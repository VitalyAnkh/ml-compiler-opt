# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local ES trainer."""

from absl import flags, logging
import enum
import functools
import gin
import tensorflow as tf
import os

# Pytype cannot pick up the pyi file for tensorflow.summary. Disable the error
# here as these errors are false positives.
# pytype: disable=pyi-error

from compiler_opt.distributed import worker_manager
from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.es import blackbox_optimizers
from compiler_opt.es import gradient_ascent_optimization_algorithms
from compiler_opt.es import blackbox_learner
from compiler_opt.es import policy_utils
from compiler_opt.rl import corpus

FLAGS = flags.FLAGS

_GRAD_REG_ALPHA = flags.DEFINE_float(
    "grad_reg_alpha", 0.01,
    "Weight of regularization term in regression gradient.")
_GRAD_REG_TYPE = flags.DEFINE_string(
    "grad_reg_type", "ridge",
    "Regularization method to use with regression gradient.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "",
                                   "Path to write all output")
_PRETRAINED_POLICY_PATH = flags.DEFINE_string(
    "pretrained_policy_path", None,
    "The path of the pretrained policy. If not provided, it will \
        construct a new policy with randomly initialized weights.")
_REQUEST_DEADLINE = flags.DEFINE_float(
    "request_deadline", 30.0, "Deadline in seconds for requests \
    to the data collection requests.")
_TRAIN_CORPORA = flags.DEFINE_string("train_corpora", "",
                                     "List of paths to training corpora")
_NUM_WORKERS = flags.DEFINE_integer("num_workers", 100,
                                    "The number of workers to create.")


@gin.constants_from_enum(module="es_trainer_lib")
class GradientAscentOptimizerType(enum.Enum):
  INVALID = 0
  MOMENTUM = enum.auto()
  ADAM = enum.auto()


@gin.configurable
def train(additional_compilation_flags=(),
          delete_compilation_flags=(),
          replace_compilation_flags=(),
          worker_class=None,
          beta1=0.9,
          beta2=0.999,
          momentum=0.0,
          gradient_ascent_optimizer_type=GradientAscentOptimizerType.ADAM,
          worker_manager_class: type[
              worker_manager.WorkerManager] = local_worker_manager
          .LocalWorkerPoolManager):
  """Train with ES."""

  if not _TRAIN_CORPORA.value:
    raise ValueError("Need to supply nonempty train corpora.")

  # Create directories
  if not tf.io.gfile.isdir(_OUTPUT_PATH.value):
    tf.io.gfile.makedirs(_OUTPUT_PATH.value)

  # Construct the policy and upload it
  policy = policy_utils.create_actor_policy()

  # Get initial parameter
  if not _PRETRAINED_POLICY_PATH.value:
    # Use randomly initialized parameters
    logging.info("Use random parameters")
    initial_parameters = policy_utils.get_vectorized_parameters_from_policy(
        policy)
    logging.info("Parameter dimension: %s", initial_parameters.shape)
    logging.info("Initial parameters: %s", initial_parameters)
  else:
    # Read the parameters from the pretrained policy
    logging.info("Reading policy parameters from %s",
                 _PRETRAINED_POLICY_PATH.value)
    # Load the policy
    pretrained_policy = tf.saved_model.load(_PRETRAINED_POLICY_PATH.value)
    initial_parameters = policy_utils.get_vectorized_parameters_from_policy(
        pretrained_policy)

  policy_parameter_dimension = (
      policy_utils.get_vectorized_parameters_from_policy(policy).shape[0])
  if policy_parameter_dimension != initial_parameters.shape[0]:
    raise ValueError("Pretrained policy dimension is incorrect")

  logging.info("Parameter dimension: %s", initial_parameters.shape)
  logging.info("Initial parameters: %s", initial_parameters)

  cps = corpus.Corpus(
      data_path=_TRAIN_CORPORA.value,
      additional_flags=additional_compilation_flags,
      delete_flags=delete_compilation_flags,
      replace_flags=replace_compilation_flags)

  # Construct policy saver
  saved_policy = policy_utils.create_actor_policy()
  policy_saver_function = functools.partial(
      policy_utils.save_policy,
      policy=saved_policy,
      save_folder=os.path.join(_OUTPUT_PATH.value, "saved_policies"))

  # Get learner config
  learner_config = blackbox_learner.BlackboxLearnerConfig()

  # the following are from Blackbox Library.
  init_current_input = initial_parameters
  init_iteration = 0
  metaparams = []  # Ignore meta params for state normalization for now
  # TODO(linzinan): delete all unused parameters.

  # ------------------ GRADIENT ASCENT OPTIMIZERS ------------------------------
  if gradient_ascent_optimizer_type == GradientAscentOptimizerType.MOMENTUM:
    logging.info("Running momentum gradient ascent optimizer")
    # You can obtain a vanilla gradient ascent optimizer by setting momentum=0.0
    # and setting step_size to the desired learning rate.
    gradient_ascent_optimizer = (
        gradient_ascent_optimization_algorithms.MomentumOptimizer(
            learner_config.step_size, momentum))
  elif gradient_ascent_optimizer_type == GradientAscentOptimizerType.ADAM:
    logging.info("Running Adam gradient ascent optimizer")
    gradient_ascent_optimizer = (
        gradient_ascent_optimization_algorithms.AdamOptimizer(
            learner_config.step_size, beta1, beta2))
  else:
    logging.info("No gradient ascent optimizer selected. Stopping.")
    return
  # ----------------------------------------------------------------------------

  # ------------------ OPTIMIZERS ----------------------------------------------
  if learner_config.blackbox_optimizer == (
      blackbox_optimizers.Algorithm.MONTE_CARLO):
    logging.info("Running ES/ARS. Filtering: %s directions",
                 str(learner_config.num_top_directions))
    blackbox_optimizer = blackbox_optimizers.MonteCarloBlackboxOptimizer(
        learner_config.precision_parameter, learner_config.estimator_type,
        learner_config.fvalues_normalization,
        learner_config.hyperparameters_update_method, metaparams, None,
        learner_config.num_top_directions, gradient_ascent_optimizer)
  elif learner_config.blackbox_optimizer == (
      blackbox_optimizers.Algorithm.TRUST_REGION):
    logging.info("Running trust region")
    tr_params = {
        "init_radius": FLAGS.tr_init_radius,
        "grow_threshold": FLAGS.tr_grow_threshold,
        "grow_factor": FLAGS.tr_grow_factor,
        "shrink_neg_threshold": FLAGS.tr_shrink_neg_threshold,
        "shrink_factor": FLAGS.tr_shrink_factor,
        "reject_threshold": FLAGS.tr_reject_threshold,
        "reject_factor": FLAGS.tr_reject_factor,
        "dense_hessian": FLAGS.tr_dense_hessian,
        "sub_termination": FLAGS.tr_sub_termination,
        "subproblem_maxiter": FLAGS.tr_subproblem_maxiter,
        "minimum_radius": FLAGS.tr_minimum_radius,
        "grad_type": FLAGS.grad_type,
        "grad_reg_type": _GRAD_REG_TYPE.value,
        "grad_reg_alpha": _GRAD_REG_ALPHA.value
    }
    for param, value in tr_params.items():
      logging.info("%s: %s", param, value)
      blackbox_optimizer = blackbox_optimizers.TrustRegionOptimizer(
          learner_config.precision_parameter, learner_config.estimator_type,
          learner_config.fvalues_normalization,
          learner_config.hyperparameters_update_method, metaparams, tr_params)
  elif learner_config.blackbox_optimizer == (
      blackbox_optimizers.Algorithm.SKLEARN_REGRESSION):
    logging.info("Running Regression Based Optimizer")
    blackbox_optimizer = blackbox_optimizers.SklearnRegressionBlackboxOptimizer(
        _GRAD_REG_TYPE.value, _GRAD_REG_ALPHA.value,
        learner_config.estimator_type, learner_config.fvalues_normalization,
        learner_config.hyperparameters_update_method, metaparams, None,
        gradient_ascent_optimizer)
  else:
    raise ValueError(
        f"Unknown optimizer: '{learner_config.blackbox_optimizer}'")

  logging.info("Initializing blackbox learner.")
  learner = blackbox_learner.BlackboxLearner(
      blackbox_opt=blackbox_optimizer,
      train_corpus=cps,
      output_dir=_OUTPUT_PATH.value,
      policy_saver_fn=policy_saver_function,
      model_weights=init_current_input,
      config=learner_config,
      initial_step=init_iteration,
      deadline=_REQUEST_DEADLINE.value)

  if not worker_class:
    logging.info("No Worker class selected. Stopping.")
    return

  logging.info("Ready to train: running for %d steps.",
               learner_config.total_steps)

  with worker_manager_class(
      worker_class,
      count=_NUM_WORKERS.value,
      worker_kwargs=dict(gin_config=gin.operative_config_str())) as pool:
    learner.set_baseline(pool)
    for _ in range(learner_config.total_steps):
      learner.run_step(pool)

  learner.flush_models()

  return learner.get_model_weights()
