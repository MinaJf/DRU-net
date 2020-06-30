import os
import time
import shutil
import numpy as np
import tensorflow as tf
from utils import util as U
from utils.eval_saver import save_str, save_img


class Trainer:
    def __init__(self, model):
        self.model = model

    # @tf.function
    def train(self,
              train_provider,
              validation_provider,
              epochs,
              batch_size,
              output_path,
              optimizer=tf.keras.optimizers.Adam(),
              learning_rate=None,
              mini_batch_size=None,
              eval_frequency=1,
              is_save_train_imgs=False,
              is_save_valid_imgs=True,
              is_rebuilt_path=True):

        if learning_rate is None:
            learning_rate = optimizer.learning_rate.numpy()
        if type(learning_rate) is not list:
            learning_rate = [learning_rate]
        if is_rebuilt_path and os.path.exists(output_path):
            shutil.rmtree(output_path, ignore_errors=True)
        iters = train_provider.size / batch_size
        assert iters > 0 and iters % 1 == 0, 'batch size {} does not match the data size {}.'.format(batch_size,
                                                                                                     train_provider.size)
        mini_batch_size = batch_size if mini_batch_size is None else mini_batch_size
        mini_iters = batch_size / mini_batch_size
        assert mini_iters > 0 and mini_iters % 1 == 0, 'mini batch size {} does not match the batch size {}.'.format(
            mini_batch_size, batch_size)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        net_args = {}
        for i in range(len(nets)):
            net_args.update({'net%d' % i: nets[i]})
        ckpt = tf.train.Checkpoint(**net_args)

        print(
            'Start training: epochs {}, learning rate {}, batch size {}, mini-batch size {}, training data {}, validation data {}.'
            .format(epochs, [str(lr) for lr in learning_rate], batch_size, mini_batch_size, train_provider.size,
                    validation_provider.size))

        train_eval_str = {}
        valid_eval_str = {}
        best_loss = float('inf')
        time_start = time.time()
        for ep in range(epochs):
            ep_time_start = time.time()
            for _ in range(int(iters)):
                grads = None
                for _ in range(int(mini_iters)):
                    feed_dict = train_provider(mini_batch_size)
                    mini_grads = self.model.get_grads(feed_dict)
                    grads = self._grads_add(grads, mini_grads)
                grads = self._grads_div(grads, mini_iters)
                if type(grads) is tuple:
                    assert len(optimizer) == len(grads), 'Number of optimizer should equal to number of networks.'
                    for gi in range(len(grads)):
                        optimizer[gi].apply_gradients(zip(grads[gi], self.model.net[gi].trainable_variables))
                else:
                    optimizer.apply_gradients(zip(grads, self.model.net.trainable_variables))
            ep_train_time = time.time() - ep_time_start
            ep_eval_time = 0
            if ep % eval_frequency == 0 or ep == epochs - 1:
                ep_train_eval = self.eval(train_provider, batch_size=mini_batch_size, print_str=False,
                                          need_imgs=is_save_train_imgs)
                ep_valid_eval = self.eval(validation_provider, batch_size=mini_batch_size, print_str=False,
                                          need_imgs=is_save_valid_imgs)
                ep_eval_time = time.time() - ep_train_time - ep_time_start
                if is_save_train_imgs:
                    save_img(ep_train_eval[1], '{}/train_imgs/'.format(output_path), ep)
                    ep_train_eval = ep_train_eval[0]
                if is_save_valid_imgs:
                    save_img(ep_valid_eval[1], '{}/valid_imgs/'.format(output_path), ep)
                    ep_valid_eval = ep_valid_eval[0]
                save_str(ep_train_eval, '{}/train_eval.txt'.format(output_path), ep)
                save_str(ep_valid_eval, '{}/valid_eval.txt'.format(output_path), ep)
                # save best ckpt
                if np.mean(ep_valid_eval['loss']) < best_loss:
                    ckpt.write(output_path + '/ckpt/best')
                    best_loss = np.mean(ep_valid_eval['loss'])

                    # time_ep_save_imgs_end = time.time()
            train_log = (
            'epoch {} ------ time cost: overall {:.1f} ------ step training {:.1f} ------ step evaluation {:.1f} ------ learning rate: {} ------'
            .format(ep, time.time() - time_start, ep_train_time, ep_eval_time, [str(lr) for lr in learning_rate]))

            if ep % eval_frequency == 0 or ep == epochs - 1:
                train_log += ('\n  train      : {}'.format(U.dict_to_str(ep_train_eval)) + \
                              '\n  validation : {}'.format(U.dict_to_str(ep_valid_eval)))

            print(train_log)
            with open(output_path + '/train_log.txt', 'a+') as f:
                f.write(train_log + '\n')

            train_eval_str = U.dict_append(train_eval_str, ep_train_eval)
            valid_eval_str = U.dict_append(valid_eval_str, ep_valid_eval)

            # TODO add early stopping and best ckpt save
            # TODO add tensorboard summary
            ckpt.write(output_path + '/ckpt/final')

        return train_eval_str, valid_eval_str

    def restore(self, ckpt_path):
        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        net_args = {}
        for i in range(len(nets)):
            net_args.update({'net%d' % i: nets[i]})
        ckpt = tf.train.Checkpoint(**net_args)
        ckpt.restore(ckpt_path)

    def eval(self, data_in, **kwargs):
        batch_size = kwargs.get('batch_size', 1)
        print_str = kwargs.get('print_str', True)
        need_imgs = kwargs.get('need_imgs', False)
        if type(data_in) is dict:
            time_start = time.time()
            eval_dict = self.model.eval(data_in, **kwargs)
            time_cost = time.time() - time_start
            if print_str:
                print('Evaluation time cost is {:.1f}'.format(time_cost))
            return eval_dict

        data_provider = data_in
        ndata = data_provider.size
        m = ndata // batch_size
        n = ndata % batch_size
        results = {}
        imgs = {}
        if print_str:
            print('Evaluate {} data:'.format(ndata))
        time_start = time.time()
        for i in range(m):
            sub_results = self.model.eval(data_provider(batch_size), **kwargs)
            sub_imgs = sub_results.pop('imgs') if 'imgs' in sub_results else None
            results = U.dict_concat(results, sub_results)
            if need_imgs and sub_imgs is not None:
                imgs = U.dict_append(imgs, sub_imgs)
            if print_str:
                print('evalated {} data'.format(batch_size * (i + 1)))
        if n > 0:
            sub_results = self.model.eval(data_provider(n), **kwargs)
            sub_imgs = sub_results.pop('imgs') if 'imgs' in sub_results else None
            results = U.dict_concat(results, sub_results)
            if need_imgs and sub_imgs is not None:
                imgs = U.dict_append(imgs, sub_imgs)
            print('evalated {} data'.format(ndata))
        if print_str:
            time_cost = time.time() - time_start
            print('Time cost is {:.1f}'.format(time_cost))
            print('  {}'.format(U.dict_to_str(results)))
        if need_imgs:
            return results, imgs
        else:
            return results

    def predict(self, data_in, batch_size=1):
        if type(data_in) is dict:
            time_start = time.time()
            predictions = self.model.predict(data_in)
            time_cost = time.time() - time_start
            print('Prediction time cost is {:.1f}'.format(time_cost))
            return predictions
        data_provider = data_in
        ndata = data_provider.size
        m = ndata // batch_size
        n = ndata % batch_size
        predictions = None
        for _ in range(m):
            sub_predictions = self.model.predict(data_provider(batch_size))
            predictions = np.concatenate((predictions, sub_predictions),
                                         0) if predictions is not None else sub_predictions
        if n > 0:
            sub_predictions = self.model.predict(data_provider(n))
            predictions = np.concatenate((predictions, sub_predictions),
                                         0) if predictions is not None else sub_predictions
        return predictions


    def _grads_add(self, grads, mini_grads):
        if grads is None:
            grads = mini_grads
        else:
            if type(grads) is not tuple:
                for i, g in enumerate(mini_grads):
                    if g is not None:
                        grads[i] += g
            else:
                for gi in range(len(grads)):
                    sub_grads = grads[gi]
                    sub_mini_grads = mini_grads[gi]
                    for i, g in enumerate(sub_mini_grads):
                        if g is not None:
                            sub_grads[i] += g
        return grads

    def _grads_div(self, grads, n):
        if n != 1:
            if type(grads) is not tuple:
                for i in range(len(grads)):
                    if grads[i] is not None:
                        grads[i] /= n
            else:
                for gi in range(len(grads)):
                    sub_grads = grads[gi]
                    for i in range(len(sub_grads)):
                        if sub_grads[i] is not None:
                            sub_grads[i] /= n
        return grads