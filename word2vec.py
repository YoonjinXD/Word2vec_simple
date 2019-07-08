import torch
import numpy as np
import random
from random import shuffle
from collections import Counter
import argparse
from huffman import HuffmanCoding

class Node:
    def __init__(self, index):
        self.index = index
        self.left = None
        self.right = None

class HuffmanTree:

    def __init__(self, codes):
        self.root = Node(0)
        self.node_count = 1
        self.buildTree(codes)

    def buildTree(self, codes):
        for code in codes:
            current_node = self.root
            for i, path in enumerate(code):
                # Terminate before the 'word' node
                if(i == len(code)-1):
                    break

                if(path == '0'):
                    # if the next node is not defined yet, define the node.
                    if(current_node.left == None):
                        current_node.left = Node(self.node_count)
                        self.node_count += 1

                    # Move to the next node
                    current_node = current_node.left

                elif(path == '1'):
                    # if the next node is not defined yet, define the node.
                    if (current_node.right == None):
                        current_node.right = Node(self.node_count)
                        self.node_count += 1

                    # Move to the next node
                    current_node = current_node.right

        # Confirmation Message
        print("Created Huffman Code Tree. The number of nodes: ", self.node_count)

def FindPath(code, tree):
    current_node = tree.root
    path_nodes = [current_node.index, ]

    for path in code:
        if (path == '0'):
            current_node = current_node.left
        elif(path == '1'):
            current_node = current_node.right
        else: pass

        if(current_node == None):
            return path_nodes
        path_nodes.append(current_node.index)


def Analogical_Reasoning_Task(embedding):
    #######################  Input  #########################
    # embedding : Word embedding (type:torch.tesnor(V,D))   #
    #########################################################

    text = open('questions-words.txt', mode='r').readlines()

    max_sim = -1
    corrects = 0
    no_word_count = 0

    print("Start Analogical Reasoning Task... with %d sentences" % len(text))

    for i, sentence in enumerate(text):

        max_sim = -1
        pred_word = None
        words = (sentence.lower()).split()
        flag = 0
        # Error Handling
        if (len(words) != 4):
            continue
        for word in words:
            if word not in embedding.keys():
                no_word_count += 1
                flag = 1
                break
        if (flag == 1): continue

        pred_vec = embedding[words[1]] - embedding[words[0]] + embedding[words[2]]

        for word, vector in embedding.items():

            cosine_sim = torch.cosine_similarity(vector, pred_vec, 0)

            if (cosine_sim > max_sim):
                max_sim = cosine_sim
                pred_word = word

        # For test
        if (pred_word == words[3]):
            corrects += 1
            print("Correct!", words, pred_word, corrects)

        # Iteration Checking
        if (i % 2000 == 0):
            print(i, "iterations =>", words, pred_word, cosine_sim, max_sim)

    print("Total Correct Answers: ", corrects, "when the # of words is ", len(text))
    print("\n[Acuuracy]: ", (corrects / len(text)), "[Words not in corpus]: ", no_word_count)



def subsampling(word_seq):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################

    words_count = Counter(word_seq)
    total_count = len(word_seq)
    words_freq = {word: count/total_count for word, count in words_count.items()}

    prob = {}

    for word in words_freq:
        prob[word] = 1 - np.sqrt(0.00001/words_freq[word])

    subsampled = [word for word in word_seq if random.random() < (1 - prob[word])]
    return subsampled

def skipgram_HS(centerWord, contextCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# contextCode : Code of a contextword (type:str)                                  #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    inputVector = inputMatrix[centerWord]
    out = outputMatrix.mm(inputVector.view(D, 1))


    out_for_loss = torch.sigmoid(out)
    for i, path in enumerate(contextCode):
        if (path == '1'):
            out_for_loss[i] = 1 - out_for_loss[i]

    loss = -torch.log(out_for_loss).sum()

    grad = torch.sigmoid(out)
    for i, path in enumerate(contextCode):
        if (path == '0'):
            grad[i] -= 1

    grad_in = grad.view(1,-1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1,-1))

    return loss, grad_in, grad_out



def skipgram_NS(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    inputVector = inputMatrix[centerWord]
    out = outputMatrix.mm(inputVector.view(D, 1))

    out_for_loss = -out
    out_for_loss[0] = -out_for_loss[0]

    loss = -torch.log(torch.sigmoid(out_for_loss)).sum()

    grad = torch.sigmoid(out)
    grad[0] -= 1

    grad_in = grad.view(1,-1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1,-1))

    return loss, grad_in, grad_out


def CBOW_HS(contextWords, centerCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# centerCode : Code of a centerword (type:str)                                    #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    inputVector = inputMatrix[contextWords].sum(0)
    out = outputMatrix.mm(inputVector.view(D,1))

    out_for_loss = torch.sigmoid(out)
    for i, path in enumerate(centerCode):
        if(path == '1'):
            out_for_loss[i] = 1 - out_for_loss[i]

    loss = -torch.log(out_for_loss).sum()

    grad = torch.sigmoid(out)
    for i, path in enumerate(centerCode):
        if (path == '0'):
            grad[i] -= 1

    grad_in = grad.view(1,-1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1,-1))

    return loss, grad_in, grad_out


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    inputVector = inputMatrix[contextWords].sum(0)
    out = outputMatrix.mm(inputVector.view(D, 1))

    out_for_loss = -out
    out_for_loss[0] = -out_for_loss[0]

    loss = -torch.log(torch.sigmoid(out_for_loss)).sum()

    grad = torch.sigmoid(out)
    grad[0] -= 1

    grad_in = grad.view(1,-1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1,-1))

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, mode="CBOW", NS=20, dimension=10000, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples", len(input_seq))
    stats = torch.LongTensor(stats)

    # Build a corresponding Huffman Tree
    tree = HuffmanTree(codes.values())

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1
            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    output_path = FindPath(codes[output], tree)
                    activated = torch.tensor(output_path).view(-1, )

                    L, G_in, G_out = CBOW_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    random_idx = torch.randint(0, len(stats), size=(NS,))
                    neg_sample = (stats.view(-1, ))[random_idx]
                    activated = torch.cat([torch.tensor([output]), neg_sample], 0)

                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

            elif mode=="SG":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    output_path = FindPath(codes[output], tree)
                    activated = torch.tensor(output_path).view(-1, )

                    L, G_in, G_out = skipgram_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    random_idx = torch.randint(0, len(stats), size=(NS, ))
                    neg_sample = (stats.view(-1,))[random_idx]
                    activated = torch.cat([torch.tensor([output]), neg_sample], 0)

                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out

                
            else:
                print("Unkwnown mode : "+mode)
                exit()
            losses.append(L.item())
            if i%100000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("Iteration:", i, "Loss : %f" %(avg_loss,))
            	losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    ns = args.ns

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k


    #Code dict for hierarchical softmax
    freqdict={}
    freqdict[0]=10
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
    codedict = HuffmanCoding().build(freqdict)

    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    words = subsampling(words)
    #Make training set
    print("build training set...")
    input_set = []
    target_set =[]
    window_size = 5
    if mode=="CBOW":
        for j in range(len(words)):
            if j<window_size:
                input_set.append([0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
            elif j>=len(words)-window_size:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)])
                target_set.append(w2i[words[j]])
            else:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
    if mode=="SG":
        for j in range(len(words)):
            if j<window_size:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
            elif j>=len(words)-window_size:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
            else:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))

    #Training section
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), codedict, freqtable, mode=mode, NS=ns, dimension=64, epoch=1, learning_rate=0.01)

    #Creating embedding dictionary
    emb_dict = {}
    for i in range(len(w2i)):
        emb_dict[i2w[i]] = emb[i]
    #Starting Analogical task
    Analogical_Reasoning_Task(emb_dict)

main()