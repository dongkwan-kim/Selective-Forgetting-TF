import tensorflow as tf
import numpy as np
from enums import MaskType


class Mask:

    def __init__(self,
                 mask_id: int,
                 alpha: float = None,
                 not_alpha: float = None,
                 mask_type: MaskType = MaskType.TARGETED):

        self.alpha = alpha or {
            MaskType.TARGETED: 0.1,
            MaskType.ADAPTIVE: 2.,
            MaskType.RELAXED: 0,
        }[mask_type]

        self.not_alpha = not_alpha or {
            MaskType.TARGETED: 0.9,
            MaskType.ADAPTIVE: - 4.0,
            MaskType.RELAXED: 0,
        }[mask_type]

        self.mask_id = mask_id
        self.mask_type = mask_type

    def get_masked_tensor(self,
                          prev_layer,
                          is_training: tf.placeholder,
                          with_residuals: bool = False):

        returns = []
        shape = prev_layer.get_shape().as_list()
        n_units = shape[-1]
        reduce_axis = list(range(len(shape) - 1))

        if self.mask_type == MaskType.RELAXED:
            with tf.variable_scope("mask", reuse=tf.AUTO_REUSE):
                mask_var = tf.get_variable("mask_var_{}".format(self.mask_id), shape=[n_units, n_units])

            reduced_prev_layer = tf.reduce_mean(tf.matmul(prev_layer, mask_var), axis=reduce_axis)
            sampled_mask = 0.5 * (tf.nn.softsign(10 * reduced_prev_layer) + 1)

            residuals = {
                "mask_var": mask_var,
            }

        elif self.mask_type == MaskType.ADAPTIVE:
            probs_of_activation_h = 0.5 * (tf.nn.softsign(self.alpha * tf.reduce_mean(prev_layer, axis=reduce_axis)
                                                          + self.not_alpha) + 1)
            sampled_mask = tf.cast(
                tf.math.greater(
                    probs_of_activation_h,
                    tf.random_uniform(probs_of_activation_h.shape, minval=0, maxval=1)),
                dtype=tf.float32,
            )
            residuals = {
                "probs_of_activation_h": probs_of_activation_h,
            }

        elif self.mask_type == MaskType.TARGETED:
            reduced_prev_layer = tf.reduce_mean(prev_layer, axis=reduce_axis)
            k = int(n_units * self.alpha)
            top_k = tf.nn.top_k(reduced_prev_layer, k=k)
            top_k_values = top_k.values
            min_top_k_value = tf.reduce_min(top_k_values)

            top_k_mask = tf.cast(tf.math.greater(
                reduced_prev_layer,
                min_top_k_value,
            ), dtype=tf.float32)

            bottom_mask = tf.cast(tf.math.greater(
                tf.random_uniform([n_units], minval=0, maxval=1),
                self.not_alpha,  # probability of dropout.
            ), dtype=tf.float32)

            sampled_mask = tf.reduce_max([top_k_mask, bottom_mask], axis=0)
            residuals = {
                "k": k,
                "top_k_mask": top_k_mask,
            }

        else:
            raise ValueError

        cond_mask = tf.cond(
            is_training,
            lambda: sampled_mask,
            lambda: tf.constant(1., shape=[n_units]),
            name="mask_{}".format(self.mask_id),
        )

        if len(shape) == 4:  # conv2d case
            # (n_units) -> (w, h, n_units)
            cond_mask = tf.broadcast_to(cond_mask, shape[1:])

        returns.append(tf.multiply(cond_mask, prev_layer))

        if with_residuals:
            returns.append({
                "cond_mask": cond_mask,
                **residuals,
            })

        return tuple(returns)

    @staticmethod
    def get_exclusive_masked_tensor(prev_layer, mask, is_training: tf.placeholder):
        return tf.multiply(
            tf.cond(
                is_training,
                lambda: 1. - mask,
                lambda: tf.constant(1., shape=[prev_layer.shape[-1]]),
            ),
            prev_layer,
        )

    @staticmethod
    def get_mask_tensor_by_idx(mask_idx) -> tf.Tensor:
        return tf.get_default_graph().get_tensor_by_name("mask_{}/Merge:0".format(mask_idx))

    @staticmethod
    def get_mask_value_by_idx(sess: tf.Session, mask_idx, feed_dict) -> np.ndarray:
        assert len(feed_dict) >= 2, "There should be at least {X: xs, is_training: True}"
        return sess.run(Mask.get_mask_tensor_by_idx(mask_idx), feed_dict=feed_dict)

    @staticmethod
    def get_mask_stats_by_idx(sess: tf.Session, mask_idx, feed_dict) -> dict:
        v = Mask.get_mask_value_by_idx(sess, mask_idx, feed_dict)
        return {
            "#nonzero": np.count_nonzero(v),
            "#total": np.prod(v.shape),
            "indices": np.argwhere(v >= 0.75).flatten(),
        }
