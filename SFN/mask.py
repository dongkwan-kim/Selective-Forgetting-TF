import tensorflow as tf
import numpy as np
from enums import MaskType


def softsign_0_to_1(x):
    return 0.5 * (tf.nn.softsign(x) + 1)


class Mask:

    def __init__(self,
                 mask_id: int,
                 alpha: float = None,
                 not_alpha: float = None,
                 hard_mask: np.ndarray = None,
                 mask_type: MaskType = MaskType.ADAPTIVE):

        self.alpha = alpha or {
            MaskType.ADAPTIVE: 5.,
            MaskType.HARD: 0.,
            MaskType.INDEPENDENT: 0.,
        }[mask_type]

        self.not_alpha = not_alpha or {
            MaskType.ADAPTIVE: - 5.0,
            MaskType.HARD: 0.,
            MaskType.INDEPENDENT: 0.,
        }[mask_type]

        self.mask_id = mask_id
        self.hard_mask = hard_mask
        self.mask_type = mask_type

    def get_masked_tensor(self,
                          prev_layer,
                          is_training: tf.placeholder = None,
                          with_residuals: bool = False):

        returns = []
        shape = prev_layer.get_shape().as_list()
        n_units = shape[-1]
        reduce_axis = list(range(len(shape) - 1))

        residuals = {}
        if self.mask_type == MaskType.ADAPTIVE:

            reduced_layer = tf.reduce_mean(prev_layer, axis=reduce_axis)
            mean_of_layer, std_of_layer = tf.nn.moments(reduced_layer, axes=[0])
            z_layer = (reduced_layer - mean_of_layer) / std_of_layer

            probs_of_activation_h = 0.55 * softsign_0_to_1(self.alpha * z_layer + self.not_alpha)

            sampled_mask = tf.cast(
                tf.math.greater(
                    probs_of_activation_h,
                    tf.random_uniform(probs_of_activation_h.shape, minval=0, maxval=1)),
                dtype=tf.float32,
            )
            residuals = {
                "probs_of_activation_h": probs_of_activation_h,
            }

            cond_mask = tf.cond(
                is_training,
                lambda: sampled_mask,
                lambda: tf.constant(1., shape=[n_units]),
                name="mask_{}".format(self.mask_id),
            )

        elif self.mask_type == MaskType.INDEPENDENT:
            sampled_mask = tf.cast(
                tf.math.greater(
                    0.5,
                    tf.random_uniform([n_units], minval=0, maxval=1),
                ), dtype=tf.float32,
            )
            cond_mask = tf.cond(
                is_training,
                lambda: sampled_mask,
                lambda: tf.constant(1., shape=[n_units]),
                name="mask_{}".format(self.mask_id),
            )

        elif self.mask_type == MaskType.HARD:
            cond_mask = self.hard_mask
            assert cond_mask.get_shape().as_list()[-1] == n_units

        else:
            raise ValueError

        if len(shape) == 4:  # conv2d case
            # (n_units) -> (w, h, n_units)
            cond_mask = tf.broadcast_to(cond_mask, shape[1:])

        returns.append(tf.multiply(cond_mask, prev_layer))

        if with_residuals:
            returns.append({
                "cond_mask": cond_mask,
                **residuals,
            })

        if len(returns) == 1:
            return returns[0]
        else:
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
