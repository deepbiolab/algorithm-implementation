import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# some helper functions


def argmax(vec):
    # return the argmax as a python int
    # 第1维度上最大值的下标
    # input: tensor([[2,3,4]]) // size = 1 x n
    # output: 2
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    # 文本序列转化为index的序列形式
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    # compute log sum exp in a numerically stable way for the forward algorithm
    # 用数值稳定的方法计算前向传播的对数和exp
    # input: tensor([[2,3,4]])
    # max_score_broadcast: tensor([[4,4,4]])
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    # 里面先做减法，减去最大值可以避免e的指数次，计算机上溢
    return max_score+torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))


START_TAG = "<s>"
END_TAG = "<e>"

# create model


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag2ix = tag2ix
        self.tagset_size = len(tag2ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            num_layers=1, bidirectional=True)

        # maps output of lstm to tog space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # matrix of transition parameters
        # entry i, j is the score of transitioning to i from j
        # tag间的转移矩阵，是CRF层的参数
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # these two statements enforce the constraint that we never transfer to the start tag
        # and we never transfer from the stop tag
        self.transitions.data[tag2ix[START_TAG], :] = -10000
        self.transitions.data[:, tag2ix[END_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim//2),
                torch.randn(2, 1, self.hidden_dim//2))

    # compute all path score
    def _forward_alg(self, feats):

        # for example: tagset = {START_TAG, tag1, tag2, tag3, END_TAG}
        # tensor([[-10000.,-10000.,-10000.,-10000.,-10000.]])
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        # All path score at START_TAG
        # tensor([[-10000.,-10000.,-10000.,0,-10000.]])
        init_alphas[0][self.tag2ix[START_TAG]] = 0 

        # initial alpha at timestamp START_TAG
        forward_var = init_alphas

        # feats: emission matrix from bilstm output, size = n*k
        # where n = len(sentence), k = len(tagsize)
        for feat in feats:
            
            # alphas_t: a array to store score on each tag j at time i
            alphas_t = []
            for next_tag in range(self.tagset_size):

                # feat[next_tag]: get emission score at tag j
                # tensor([3]) -> tensor([[3,3,3,3,3]])
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)

                # transitions[next_tag]: get transition scores from j' to j
                trans_score = self.transitions[next_tag].view(1, -1)

                # compute alpha_{i-1, j'} + T_{j', j} + X_{j, Wi}
                next_tag_var = forward_var + trans_score + emit_score

                # compute log_sum_exp on each tag j at time i and append to alphas_t
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            
            # get all path score at time i for each tag j
            forward_var = torch.cat(alphas_t).view(1, -1)

        # get all path score at last time i (tag=END_TAG) for each tag j
        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]

        # get final all path score using log_sum_exp
        # alpha = S_{allpath}
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # compute real path score
    def _score_sentence(self, feats, tags):
        """gives the score of a provides tag sequence
        # feats: emission matrix from bilstm output, size = n*k
            # where n = len(sentence), k = len(tagsize)
        # tags: true label index at each timestamp
        """

        score = torch.zeros(1)

        # Put START_TAG at tag sequence head, such as [START_TAG， tag1, tag2... tagN]
        tags = torch.cat(
            [torch.tensor([self.tag2ix[START_TAG]], dtype=torch.long), tags])

        # Compute real path score : 
        # realpath score = each timestamp emission score + each timestamp transition score
        for i, feat in enumerate(feats):
            # transition score from i -> i+1: self.transitions[tags[i + 1], tags[i]]: 
            # emission score at i: feat[tags[i+1]], because START_TAG in tag sequence, index not i
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        
        # Add value from last tag to END_TAG at score
        score = score + self.transitions[self.tag2ix[END_TAG], tags[-1]]
        return score
    
    # Compute best path score and best path
    def _viterbi_decode(self, feats):

        backpointers = []

        # tensor([[-10000.,-10000.,-10000.,-10000.,-10000.]])
        init_vars = torch.full((1, self.tagset_size), -10000.)
        # tensor([[-10000.,-10000.,-10000.,0,-10000.]])
        init_vars[0][self.tag2ix[START_TAG]] = 0

        forward_var = init_vars
        for feat in feats:
            bptrs_t = []  # holds the back pointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # Compute loss function
    def neg_log_likelihood(self, sentence, tags):
        """
        sentence: token index at each timestamp
        tags: true label index at each timestamp
        """
        # Emission Matrix: feats, size=n*k, where n = len(sentence), k = len(tagsize)
        feats = self._get_lstm_features(sentence)

        # Real path score
        gold_score = self._score_sentence(feats, tags)

        # All path score
        forward_score = self._forward_alg(feats)
        
        # loss = - (S_realpath - S_allpath)
        loss = - (gold_score - forward_score) 
        return loss

    # 模型inference逻辑
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__ == "__main__":
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word2ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word2ix:
                word2ix[word] = len(word2ix)

    tag2ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4}

    model = BiLSTM_CRF(len(word2ix), tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    # 输出训练前的预测序列
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        precheck_tags = torch.tensor(
            [tag2ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word2ix)
            targets = torch.tensor([tag2ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        print(model(precheck_sent))

    # 输出结果
    # (tensor(-9996.9365), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    # (tensor(-9973.2725), [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
