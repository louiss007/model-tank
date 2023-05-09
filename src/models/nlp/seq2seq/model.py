"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 23-1-23 下午5:36
# @FileName: model.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import os
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import pickle
import numpy as np
import random
from src.models.nlp.seq2seq.sample import Sample


class Model(object):
    """ Base Class of Seq2Seq Model """
    def __init__(self, args, task_type):
        self.spec_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.tgt_time_step = args.tgt_time_step
        self.start_token_id = args.start_token_id
        self.end_token_id = args.end_token_id

        if self.src_vocab_size is None or self.tgt_time_step is None or self.tgt_vocab_size is None \
                or self.start_token_id is None or self.end_token_id is None:
            print("some of given training parameters are None!")
            exit(0)

        self.src_embedding_size = args.src_embedding_size
        self.tgt_embedding_size = args.tgt_embedding_size

        self.loss = None
        self.train_op = None
        self.task_type = task_type

        # io paras
        self.input = args.input
        self.output = args.output

        # hyper paras of model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.lr = args.lr
        # because encoder and decoder are different layers and num units, not need predefine the two paras
        # self.layer_num = args.layer_num
        # self.num_units = args.num_units

        self.sample = Sample(os.path.join(self.input, 'src.txt'), os.path.join(self.input, 'tgt.txt'))
        self.src_batch = None
        self.embedded_src_batch = None
        self.tgt_batch_x = None
        self.tgt_batch_y = None
        self.tgt_embedding = None
        self.embedded_tgt_batch_x = None
        self.src_batch_seq_len = None
        self.tgt_batch_seq_len = None
        self.tgt_batch_max_len = None
        self.start_tokens = None
        self.decoder_rnn_layer = None
        self.projection_layer = None
        # for predict
        self.loaded_graph = None
        self.sess = None

    def build_input_tensor(self):
        with tf.name_scope("input_tensor"):
            # 这里设置位None，意思是不指定batch_size用到多少就是多少
            self.src_batch = tf.placeholder(tf.int32, [None, None], name="source_batch")
            # 对source做词嵌入，词向量矩阵的shape为[source的词库大小,嵌入维度]
            # 嵌入矩阵中每一行就是一个词向量
            src_embedding = tf.get_variable(shape=[self.src_vocab_size, self.src_embedding_size],
                                            name='source_embedding')
            # 使用embedding_lookup从embedding矩阵中查询词向量从而将X的每一个单词的index转换一个词向量
            self.embedded_src_batch = tf.nn.embedding_lookup(src_embedding, self.src_batch)
            # 最后返回的embedded_X.shape = [time_step,batch_size,embedding_size]
            # target类似
            self.tgt_batch_x = tf.placeholder(tf.int32, [None, None], name="target_batch_x")
            self.tgt_batch_y = tf.placeholder(tf.int32, [None, None], name="target_batch_y")
            self.tgt_embedding = tf.get_variable(shape=[self.tgt_vocab_size, self.tgt_embedding_size],
                                                 name='target_embedding')
            self.embedded_tgt_batch_x = tf.nn.embedding_lookup(self.tgt_embedding, self.tgt_batch_x)
            self.src_batch_seq_len = tf.placeholder(tf.int32, [None], name="source_batch_seq_len")
            self.tgt_batch_seq_len = tf.placeholder(tf.int32, [None], name="target_batch_seq_len")
            # 保存当前batch的最长序列值，mask的时候需要用到
            self.tgt_batch_max_len = tf.placeholder(tf.int32, [], name="target_batch_max_len")
            # 测试时输入数据的batch是多少，动态的传入，避免预测时必须固定batch
            test_batch_size = tf.placeholder(dtype=tf.int32, shape=[1], name="input_batch_size")
            # tf.tile将常量重复shape次数连接在一起
            self.start_tokens = tf.tile(tf.constant(value=self.start_token_id, dtype=tf.int32, shape=[1]),
                                        multiples=test_batch_size, name="start_tokens")

    # def create_rnn_layer(self, layer_size, num_units, cell_type="LSTM"):
    #     # create rnn layer
    #     def create_rnn_cell():
    #         if cell_type.lower() == "lstm":
    #             return tf.nn.rnn_cell.LSTMCell(num_units=num_units)
    #         elif cell_type.lower() == "basiclstm":
    #             return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
    #         elif cell_type.lower() == "gru":
    #             return tf.nn.rnn_cell.GRUCell(num_units=num_units)
    #         else:
    #             print("there is no this kind of cell type!")
    #             exit(0)
    #     rnn_layers = tf.nn.rnn_cell.MultiRNNCell([create_rnn_cell() for _ in range(layer_size)])
    #     return rnn_layers
    def create_rnn_layer(self, layer_size, num_units, cell_type="LSTM"):
        # create rnn layer
        def create_rnn_cell():
            if cell_type.lower() == "lstm":
                return tf.nn.rnn_cell.LSTMCell(num_units=num_units)
            elif cell_type.lower() == "basiclstm":
                return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
            elif cell_type.lower() == "gru":
                return tf.nn.rnn_cell.GRUCell(num_units=num_units)
            else:
                print("there is no this kind of cell type!")
                exit(0)
        rnn_layers = tf.nn.rnn_cell.MultiRNNCell([create_rnn_cell() for _ in range(layer_size)])
        return rnn_layers

    def build_encoder(self, encoder_layer_size, encoder_num_units, cell_type="LSTM"):
        # 定义encoder
        with tf.name_scope("encoder"):
            rnn_layer = self.create_rnn_layer(encoder_layer_size, encoder_num_units, cell_type)
            # 将rnn沿时间序列展开
            #   encoder_outputs: [batch_size,time_step, num_units]
            #   encoder_state: [batch_size, num_units]
            #   sequence_length:传入一个list保存每个样本的序列的真实长度，教程中说这样做有助于提高效率
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                rnn_layer, self.embedded_src_batch,
                sequence_length=self.src_batch_seq_len, time_major=False, dtype=tf.float32)
            # 注意返回的encoder_outputs返回了每一个序列节点的output shape为[max_time, batch_size, num_units]
            print("encoder_states:", encoder_states)
            print("encoder_outputs:", encoder_outputs)
            print("len(encoder_states):", len(encoder_states))
            # 关于encoder_states：一个tuple，有多少个cell，元组的size就是多少，保存了每一个cell运行后的c和h值
        return encoder_outputs, encoder_states

    """双向rnn的encoder
    参照Google官方的nmt教程
    """
    def build_bi_encoder(self, encoder_layer_size, encoder_num_units, cell_type="LSTM"):
        cell_type = cell_type.lower()
        # 定义bi_encoder
        with tf.name_scope("bi_encoder"):
            fw_rnn_layer = self.create_rnn_layer(encoder_layer_size, encoder_num_units, cell_type)
            bw_rnn_layer = self.create_rnn_layer(encoder_layer_size, encoder_num_units, cell_type)
            # 双向rnn展开
            '''
            bi_state的结构:(fw_state,bw_state) fw_state=((c,h),(c,h)...) bw_state = ((c,h),(c,h)...)
            '''
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn_layer, bw_rnn_layer, self.embedded_src_batch,
                sequence_length=self.src_batch_seq_len, time_major=False, dtype=tf.float32)
            '''                    
            将前向rnn和后向rnn的output的最后一个维度连接起来,比如 fw:[128,10,100],bw:[128,10,100],那么连接后为[128,10,200]
            这样导致一个问题就是encoder rnn的输出和decoder rnn的输入对应不上了，encoder因为拼接了fw和bw变成了200，
            有  2个解决办法，将encoder rnn的num_units变成decoder的一半或者反过来将decoder rnn的num_units增大一倍
            '''
            encoder_outpus = tf.concat(bi_outputs, -1)
            '''
            bi_state同样有fw和bw，怎样拼接在一起呢？
            (1)参照output的拼接直接，拼接c和h的最后一个维度：这种方法问题在于tuple必须是特殊类型的tuple,比如LSTMStateTuple
            (2)直接将fw和bw的结果堆叠在一起这样cell的个数相当于翻了一倍，需要调整encoder或者decoder的cell个数
            另外使用不同的cell也有区别的，LSTM有c和h，而GRU只有一个值。
            '''
            fw_encoder_state = bi_state[0]
            bw_encoder_state = bi_state[1]
            encoder_states = []
            if cell_type == "lstm" or cell_type == "basiclstm":
                # i循环cell的个数
                for i in range(encoder_layer_size):
                    # 连接当前cell fw和bw的c，h
                    c = tf.concat([fw_encoder_state[i][0], bw_encoder_state[i][0]], -1)
                    h = tf.concat([fw_encoder_state[i][1], bw_encoder_state[i][1]], -1)
                    encoder_states.append(LSTMStateTuple(c, h))
            else:  # GRU
                # state中每个cell只有一个值
                for i in range(encoder_layer_size):
                    state = tf.concat([fw_encoder_state[i], bw_encoder_state[i]], -1)
                    encoder_states.append(state)
            encoder_states = tuple(encoder_states)
        print("bidirectional encoder-encoder_outputs:", encoder_outpus)
        print("bidirectional encoder-encoder_states:", encoder_states)
        return encoder_outpus, encoder_states

    def build_decoder(self, encoder_states, decoder_layer_size, decoder_num_units, cell_type="LSTM"):
        # define decoder
        with tf.name_scope("decoder"):
            self.decoder_rnn_layer = self.create_rnn_layer(decoder_layer_size, decoder_num_units, cell_type)
        # decoder：需要helper，rnn cell，helper被分离开可以使用不同的解码策略，比如预测时beam search，贪婪算法
        # 这里projection_layer就是一个全连接层，encoder_output的维度不能和target词汇数量一致所以需要映射层
        with tf.name_scope("decoder_porjection"):
            # 为什么这里projection_layer不指定激活函数为softmax，最后构建loss的传入的是logits，我的理解logits是没有经过激活函数的
            # 的值，logits = W*X+b
            self.projection_layer = tf.layers.Dense(units=self.tgt_vocab_size,
                                                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                       stddev=0.1))
        # 训练使用的training_helper
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            self.embedded_tgt_batch_x, self.tgt_batch_seq_len, time_major=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_rnn_layer, training_helper, encoder_states,
            output_layer=self.projection_layer)
        # 将序列展开
        '''
        impute_finished=False,
        maximum_iterations=None,
        swap_memory=False , Whether GPU-CPU memory swap is enabled for this loop.
        '''
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False, swap_memory=True)
        # decoder_output的shape:[batch_size,序列长度,tgt_vocab_size]
        print("decoder_output:", decoder_output)
        return decoder_output

    def inference_greedy(self, encoder_states):
        # 这部分放到input_tensor中
        '''
        # 测试时输入数据的batch是多少，动态的传入，避免预测时必须固定batch
        inuput_batch = tf.placeholder(dtype=tf.int32, shape=[1],name="input_batch_size")
        # tf.tile将常量重复shape次数连接在一起
        start_tokens = tf.tile(tf.constant(value=self.start_token_id, dtype=tf.int32,shape=[1]), multiples = inuput_batch, name="start_tokens")
        '''
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.tgt_embedding, self.start_tokens,
                                                                     self.end_token_id)
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_rnn_layer, predicting_helper, encoder_states,
            output_layer=self.projection_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, output_time_major=False,
                                                                            maximum_iterations=self.tgt_time_step,
                                                                            swap_memory=True)
        # 标识预测结果tensor名字
        tf.identity(predicting_decoder_output.sample_id, name='predictions')

    # 使用beam search做inference
    def inference_beam_search(self, beam_width, encoder_states):
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            encoder_states, multiplier=beam_width)
        # 声明beam search的decoder
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decoder_rnn_layer,
            embedding=self.tgt_embedding,
            start_tokens=self.start_tokens,
            end_token=self.end_token_id,
            initial_state=decoder_initial_state,
            beam_width=beam_width,
            output_layer=self.projection_layer,
            length_penalty_weight=0.0,
            coverage_penalty_weight=0.0
        )

        # Dynamic decoding
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False,
                                                                            maximum_iterations=self.tgt_time_step,
                                                                            swap_memory=True)
        # predicted_ids：返回shape:[batchsize,max_len,beam_width]
        # predicted_ids保存了所有结果，从好到坏排名，这里直接取最好的
        tf.identity(predicting_decoder_output.predicted_ids[:, :, 0], name='predictions')

    # 构造计算图，包括loss，优化等
    def build_model(self, layer_size=2, num_units=128, cell_type="LSTM", is_bidirectional=False, decoding_method="greedy", beam_width=10):
        decoding_method = decoding_method.lower()
        # 构造计算图
        train_graph = tf.Graph()
        with train_graph.as_default():
            # 1.tensor声明
            self.build_input_tensor()
            # 2.encoder decoder inference
            if is_bidirectional:
                # 需要注意使用bidirectional，encoder rnn的num_units变为decoder的一半，这是为了能够保证encoder_states和decoder的输入shape能对应上
                encoder_outputs, encoder_states = self.build_bi_encoder(layer_size, num_units / 2, cell_type)
            else:
                encoder_outputs, encoder_states = self.build_encoder(layer_size, num_units, cell_type)
            decoder_outputs = self.build_decoder(encoder_states, layer_size, num_units, cell_type)
            # 选择合适的解码方法
            if decoding_method == "greedy":
                self.inference_greedy(encoder_states)
            elif decoding_method == 'beamsearch':
                if beam_width <= 1:
                    print("the beam width must be greater than 1! the default beam width '10' will be used!")
                    beam_width = 10
                self.inference_beam_search(beam_width, encoder_states)
            else:
                print("no such decoding method! the default method 'greedy' will be used")
                self.inference_greedy(encoder_states)
            # 1和2这个步骤必须在同一个graph下声明
            # 对训练输出取名字
            training_logits = tf.identity(decoder_outputs.rnn_output, 'logits')
            print("training_logits.shape:", training_logits.shape)
            print("tgt_batch_y.shape:", self.tgt_batch_y.shape)
            # 尝试是否能成功
            # mask的作用是：计算loss时忽略pad的部分，这部分的loss不需要算，
            masks = tf.sequence_mask(self.tgt_batch_seq_len, self.tgt_batch_max_len, dtype=tf.float32, name='masks')
            with tf.name_scope("optimization"):
                # loss
                self.loss = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    self.tgt_batch_y,
                    masks)
                # self.loss = self.seq_loss(training_logits)
                optimizer = tf.train.AdamOptimizer(self.lr)
                # optimizer = tf.train.GradientDescentOptimizer(lr)
                # Gradient Clipping
                gradients = optimizer.compute_gradients(self.loss)
                capped_gradients = [
                    (tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None
                ]
                self.train_op = optimizer.apply_gradients(capped_gradients)
        return train_graph

    # 训练
    def train(self, generator, layer_size=2, num_units=128, cell_type="LSTM", is_bidirectional=False, decoding_method="greedy", beam_width=10):
        model_graph = self.build_model(layer_size, num_units, cell_type, is_bidirectional, decoding_method, beam_width)
        with tf.Session(graph=model_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for src_batch, tgt_batch_x, tgt_batch_y, tgt_batch_max_len, src_batch_seq_len, tgt_batch_seq_len, batch_num, epoch_num in generator:
                seq_loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                    self.src_batch: src_batch,
                    self.tgt_batch_x: tgt_batch_x,
                    self.tgt_batch_y: tgt_batch_y,  # np.array(tgt_batch_y).reshape(target_max_len,batch_size)
                    self.src_batch_seq_len: src_batch_seq_len,
                    self.tgt_batch_seq_len: tgt_batch_seq_len,
                    self.tgt_batch_max_len: tgt_batch_max_len
                })
                print("epoch:", epoch_num, " batch:", batch_num, "loss:", seq_loss)
            # 保存模型
            self.save_model(sess)

    # 预测
    # input: [batch_size,time_step]
    # input_seq_len:[batch_size]
    def predict(self, input_seq, input_seq_len):
        source_batch_input = self.loaded_graph.get_tensor_by_name('input_tensor/source_batch:0')
        logits = self.loaded_graph.get_tensor_by_name('predictions:0')
        input_batch_size = self.loaded_graph.get_tensor_by_name("input_tensor/input_batch_size:0")
        src_seq_len = self.loaded_graph.get_tensor_by_name('input_tensor/source_batch_seq_len:0')
        answer_logits = self.sess.run(
            logits,
            {
                source_batch_input: input_seq,
                src_seq_len: input_seq_len,
                input_batch_size: [len(input_seq)]
            }
        )
        self.sess.close()
        return answer_logits

    def save_model(self, sess):
        saver = tf.train.Saver()
        saver.save(
            sess,
            save_path=os.path.join(self.output, self.__class__.__name__.lower()) + '.ckpt',
            # global_step=self.global_step
        )

    def load_model(self):
        # 加载模型
        self.loaded_graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.loaded_graph)
        # self.sess = tf.Session(graph=self.loaded_graph)
        # 加载模型
        loader = tf.train.import_meta_graph(self.output + '.meta')
        loader.restore(self.sess, self.output)

    @staticmethod
    def load_data(in_path):
        fi1 = open(os.path.join(in_path, 'src_id_seq_list.pkl'), 'rb')
        fi2 = open(os.path.join(in_path, 'src_id_seq_len_list.pkl'), 'rb')
        fi3 = open(os.path.join(in_path, 'tgt_id_seq_list.pkl'), 'rb')
        fi4 = open(os.path.join(in_path, 'tgt_id_seq_len_list.pkl'), 'rb')
        src_id_seq_list = pickle.load(fi1)
        src_id_seq_len_list = pickle.load(fi2)
        tgt_id_seq_list = pickle.load(fi3)
        tgt_id_seq_len_list = pickle.load(fi4)
        return src_id_seq_list, src_id_seq_len_list, tgt_id_seq_list, tgt_id_seq_len_list

    def make_train_batch(self, in_path):
        src_id_seq_list, src_id_seq_len_list, tgt_id_seq_list, tgt_id_seq_len_list = self.load_data(in_path)
        num_sample = len(src_id_seq_list)
        num_batch = num_sample // self.batch_size

        indices = [i for i in range(num_sample)]
        src_id_seq_list = np.array(src_id_seq_list)
        tgt_id_seq_list = np.array(tgt_id_seq_list)
        for i in range(self.epochs):
            random.shuffle(indices)
            for j in range(num_batch):
                src_seq_batch = [src_id_seq_list[index] for index in indices[j * self.batch_size:(j + 1) * self.batch_size]]
                src_seq_len_batch = [
                    src_id_seq_len_list[index] for index in indices[j * self.batch_size:(j + 1) * self.batch_size]
                ]
                src_seq_max_len = np.max(src_seq_len_batch)
                # 丢掉source_batch不必要的部分
                src_seq_batch = np.array(src_seq_batch)[:, 0:src_seq_max_len]

                '''
                x:<s> A B C D  <pad> <pad>
                y: A  B C D </s> <pad> <pad>
                '''
                tgt_x_batch = [tgt_id_seq_list[index] for index in indices[j * self.batch_size:(j + 1) * self.batch_size]]
                # 将tgt_x_batch向前移动一格，末尾补充pad，也就是用上一个字预测下一个字
                tgt_y_batch = [[tgt_id_seq_list[index, k] for k in range(1, tgt_id_seq_len_list[index])] for index in
                                  indices[j * self.batch_size:(j + 1) * self.batch_size]]
                for seq in tgt_y_batch:
                    # 每个seq后面添加</s>
                    seq.append(self.sample.tgt_token2id['<EOS>'])
                    # 剩余位置添加<pad>
                    ty_len = len(seq)
                    for _ in range(self.sample.tgt_max_len - ty_len):
                        seq.append(self.sample.tgt_token2id['<PAD>'])

                tgt_seq_len_batch = [
                    tgt_id_seq_len_list[index] for index in indices[j * self.batch_size:(j + 1) * self.batch_size]
                ]
                tgt_seq_max_len = np.max(tgt_seq_len_batch)
                # 同样丢掉target中不必要的部分
                tgt_x_batch = np.array(tgt_x_batch)[:, 0:tgt_seq_max_len]
                tgt_y_batch = np.array(tgt_y_batch)[:, 0:tgt_seq_max_len]
                yield src_seq_batch, tgt_x_batch, tgt_y_batch, tgt_seq_max_len, src_seq_len_batch, tgt_seq_len_batch, \
                      str(j + 1) + '/' + str(num_batch), str(i + 1) + '/' + str(self.epochs)
