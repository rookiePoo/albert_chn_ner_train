import pickle
import modeling
import tokenization
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class BertNerDataProcessor:
    def __init__(self, max_seq_len, vocab_file, do_lower_case=False):
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_len = max_seq_len

    def process_lines(self, lines):
        examples = self._create_example(lines)
        tokens, input_ids, input_masks, segment_ids, labels = [], [], [], [], []
        for (ex_idx, example) in enumerate(examples):
            #print(ex_idx ,example)
            if ex_idx % 100 == 0:
                print('converting sample %d of %d' % ( ex_idx, len(examples)))
            token, input_id, input_mask, segment_id, label = self._convert_single_example(example, self.max_seq_len, self.tokenizer)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)
            tokens.append(token)
        return tokens, input_ids, input_masks, segment_ids,labels


    def _create_example(self, lines, set_type='pred'):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line)
            #labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts))
        return examples

    def _convert_single_example(self, example, max_seq_len, tokenizer):
        textlist = example.text.split(' ')
        tokens = []

        for i, word in enumerate(textlist):
            #print(i, word)
            token = tokenizer.tokenize(word)
            tokens.extend(token)

        # only Account for [CLS] with "- 1".
        if len(tokens) >= max_seq_len - 1:
            tokens = tokens[0:(max_seq_len - 2)]

        ntokens = []
        segment_ids = []

        ntokens.append("[CLS]")
        segment_ids.append(0)
        #label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            #label_ids.append(label_map[labels[i]])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        # use zero to padding and you should
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            ntokens.append("**NULL**")
        label_seq = [0] * max_seq_len
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label_ids: %s" % " ".join([str(x) for x in label_seq]))

        return ntokens, input_ids, input_mask, segment_ids, label_seq


class BERT_CRF():
    def __init__(self, ):
        self.bert_config = modeling.BertConfig.from_json_file("albert_base_zh/albert_config_base.json")
        with open('albert_base_ner_checkpoints/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            self.id2label = {value: key for key, value in label2id.items()}
            print(self.id2label)
        self.max_seq_length = 128
        self.input_ids = tf.placeholder(shape=[None, self.max_seq_length], dtype=tf.int64, name="input_ids")
        self.input_mask = tf.placeholder(shape=[None, self.max_seq_length], dtype=tf.int64, name="input_mask")
        self.segment_ids = tf.placeholder(shape=[None, self.max_seq_length], dtype=tf.int64, name="segment_ids")

    def get_labels(self):
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        return ["O", 'B-PRICE', 'I-PRICE', 'B-BIDDER', 'I-BIDDER', 
                'B-SDATE', 'I-SDATE', 'B-TENDER', 'I-TENDER',"[CLS]","[SEP]"]
    def create_model(self, ):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        labels = self.get_labels()
        num_labels = len(labels) + 1
        output_layer = model.get_sequence_output()

        hidden_size = output_layer.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, self.max_seq_length, num_labels])

            # log_probs = tf.nn.log_softmax(logits, axis=-1)
            # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = tf.argmax(probabilities, axis=-1)
            return logits, predict

    def load_model(self, model_dir):
        configsession = tf.ConfigProto()
        # configsession.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configsession)
        with self.sess.as_default():
            self.logits, self.pred_id = self.create_model()

            saver = tf.train.Saver()
            saver.restore(self.sess, model_dir)
            # 存为pb
            constant_graph = convert_variables_to_constants(self.sess, self.sess.graph_def, ['loss/ArgMax'])
            # 写入序列化的 PB 文件
            with tf.gfile.FastGFile(model_dir + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            self.nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
            # for node in self.nodes:
            #     print(node)

    def predict(self, input_ids, input_masks, segment_ids, tokens):
#         for nname in self.nodes:
#             print(nname)
        with self.sess.graph.as_default():
            with self.sess.as_default():
                feed_dict = {self.input_ids: input_ids,
                             self.input_mask: input_masks,
                             self.segment_ids: segment_ids}

                result = self.sess.run([self.pred_id], feed_dict=feed_dict)
                # probabilities = tf.nn.softmax(log_probs, axis=-1)
                # predict = tf.argmax(probabilities, axis=-1)

                for prediction in result[0]:
                    print(prediction)
                    output_line = "\n".join(self.id2label[id] for id in prediction if id != 0) + "\n"
                    print(output_line)
                #print(result)
                #self.combineWordPieceAndLabels(tokens, log_probs[0])

        return result
if __name__ == "__main__":
    vocab_file = 'albert_config/vocab.txt'
    max_seq_length = 128

    bndp = BertNerDataProcessor(max_seq_length, vocab_file)
    abert = BERT_CRF()
    abert.load_model('albert_base_ner_checkpoints/model.ckpt-2649')

    