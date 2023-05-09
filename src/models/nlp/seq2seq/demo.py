"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 23-1-14 下午6:21
# @FileName: demo.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import numpy as np


# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001


def get_data(source_file, target_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = f.read()

    with open(target_file, 'r', encoding='utf-8') as f:
        target_data = f.read()
    return source_data, target_data


def build_vocab_map(data):
    """
    构造映射表
    :param data:
    :return:
    """
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    set_words = list(set([character for line in data.split('\n') for character in line]))
    # 这里要把四个特殊字符添加进词典
    ind2word = {idx: word for idx, word in enumerate(special_words + set_words)}
    word2ind = {word: idx for idx, word in ind2word.items()}

    return ind2word, word2ind


def data_preprocess(source_data, target_data):
    # 对字母进行转换
    source_data_ids = [[s_word2ind.get(letter, s_word2ind['<UNK>'])
                        for letter in line] for line in source_data.split('\n')]
    target_data_ids = [[t_word2ind.get(letter, t_word2ind['<UNK>'])
                        for letter in line] + [t_word2ind['<EOS>']] for line in target_data.split('\n')]

    return source_data_ids, target_data_ids


def get_inputs():
    """
    模型输入tensor
    :return:
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    """
    构造Encoder层

    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    """
    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                      sequence_length=source_sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


def process_decoder_input(data, vocab_to_int, batch_size):
    """
    补充<GO>，并移除最后一个字符
    :param data:
    :param vocab_to_int:
    :param batch_size:
    :return:
    """
    # cut掉最后一个字符
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    """
    构造Decoder层

    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    """
    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 2. 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # 3. Output全连接层
    output_layer = tf.layers.Dense(
        target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
    )

    # 4. Training decoder
    with tf.variable_scope("decode"):
        # 得到help对象
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # 构造decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
    # 5. Predicting decoder
    # 与training共享参数
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     target_letter_to_int['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers):
    # 获取encoder的状态输出
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    # 预处理后的decoder输入
    decoder_input = process_decoder_input(targets, t_word2ind, batch_size)

    # 将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(t_word2ind,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)

    return training_decoder_output, predicting_decoder_output


def build_graph():
    # 构造graph
    train_graph = tf.Graph()

    with train_graph.as_default():
        # 获得模型输入
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

        training_decoder_output, predicting_decoder_output = seq2seq_model(
            input_data,
            targets,
            lr,
            target_sequence_length,
            max_target_sequence_length,
            source_sequence_length,
            len(s_word2ind),
            len(t_word2ind),
            encoding_embedding_size,
            decoding_embedding_size,
            rnn_size,
            num_layers
        )

        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            seq_loss = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(seq_loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    return train_graph, seq_loss, train_op


def pad_sentence_batch(sentence_batch, pad_int):
    """
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    """
    定义data生成器，用来获取batch
    :param targets:
    :param sources:
    :param batch_size:
    :param source_pad_int:
    :param target_pad_int:
    :return:
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


def train():
    # 构造graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        # 获得模型输入
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

        training_decoder_output, predicting_decoder_output = seq2seq_model(
            input_data,
            targets,
            lr,
            target_sequence_length,
            max_target_sequence_length,
            source_sequence_length,
            len(s_word2ind),
            len(t_word2ind),
            encoding_embedding_size,
            decoding_embedding_size,
            rnn_size,
            num_layers
        )

        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            seq_loss = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(seq_loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    # 将数据集分割为train和validation
    train_source = source_data_ids[batch_size:]
    train_target = target_data_ids[batch_size:]
    # 留出一个batch进行验证
    valid_source = source_data_ids[:batch_size]
    valid_target = target_data_ids[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
        get_batches(valid_target, valid_source, batch_size,
                    s_word2ind['<PAD>'],
                    t_word2ind['<PAD>']))

    display_step = 50  # 每隔50轮输出loss

    checkpoint = "trained_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                                s_word2ind['<PAD>'],
                                t_word2ind['<PAD>'])):

                _, loss = sess.run(
                    [train_op, seq_loss],
                    {
                        input_data: sources_batch,
                        targets: targets_batch,
                        lr: learning_rate,
                        target_sequence_length: targets_lengths,
                        source_sequence_length: sources_lengths
                    }
                )

                if batch_i % display_step == 0:
                    # 计算validation loss
                    validation_loss = sess.run(
                        [seq_loss],
                        {
                            input_data: valid_sources_batch,
                            targets: valid_targets_batch,
                            lr: learning_rate,
                            target_sequence_length: valid_targets_lengths,
                            source_sequence_length: valid_sources_lengths
                        }
                    )

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(train_source) // batch_size,
                                  loss,
                                  validation_loss[0]))

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')


def source_to_seq(text):
    """
    对源数据进行转换
    :param text:
    :return:
    """
    sequence_length = 7
    return [s_word2ind.get(word, s_word2ind['<UNK>']) for word in text] + \
           [s_word2ind['<PAD>']] * (sequence_length - len(text))


def predict():
    # 输入一个单词
    input_word = 'common'
    text = source_to_seq(input_word)

    checkpoint = "./trained_model.ckpt"

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # 加载模型
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('inputs:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          target_sequence_length: [len(input_word)] * batch_size,
                                          source_sequence_length: [len(input_word)] * batch_size})[0]

    pad = s_word2ind["<PAD>"]

    print('原始输入:', input_word)

    print('\nSource')
    print('  Word 编号:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([s_ind2word[i] for i in text])))

    print('\nTarget')
    print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(" ".join([t_ind2word[i] for i in answer_logits if i != pad])))


if __name__ == '__main__':
    source_file = 'data/nlp/seq2seq/src.txt'
    target_file = 'data/nlp/seq2seq/tgt.txt'
    source_data, target_data = get_data(source_file, target_file)
    # 构造映射表
    s_ind2word, s_word2ind = build_vocab_map(source_data)
    t_ind2word, t_word2ind = build_vocab_map(target_data)
    source_data_ids, target_data_ids = data_preprocess(source_data, target_data)
    train()

