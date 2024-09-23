import torch,numpy as np,os,random
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch.nn as nn,logging,argparse
from CTCC_data_loader import get_task_processor
from transformers import AutoModel,AdamW,BertTokenizer,BertForSequenceClassification
from tqdm import tqdm, trange
import torch.utils.data as Data,json
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score,precision_score,recall_score

T=0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

crition_mse = nn.MSELoss(reduction='elementwise_mean').cuda()
crition_ce = nn.CrossEntropyLoss().cuda()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

flag_embding = 4
lamb=25
class InputFeatures(object):
    def __init__(self, input_ids, lables, token_type_ids, attention_mask):
        self.input_ids = input_ids
        self.lables = lables
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

class errorFeatures(object):

    def __init__(self, input_ids, lables, token_type_ids, attention_mask,confidence):
        self.input_ids = input_ids
        self.lables = lables
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.confidence = confidence

def one_hot(x, class_count):
	return torch.eye(class_count)[x,:]

class MyModel(nn.Module):
    def __init__(self,model_name='bert-base-chinese/',hidden_size=768, num_classes=819):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name,output_hidden_states=True, return_dict=True)  

        if flag_embding == 4:
            self.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(hidden_size * 4, num_classes, bias=False)  
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(hidden_size, num_classes, bias=False)  
            )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,flag_embding=None):
        if flag_embding == 1:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)
            hidden_states = outputs.hidden_states[-1]  
            first_hidden_states = hidden_states[:, 0, :]  
            logit = self.fc(first_hidden_states)  
        else:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)
            hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]),
                                        dim=-1)  
            first_hidden_states = hidden_states[:, 0, :]  
            logit = self.fc(first_hidden_states)  

        return logit 

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, seed=12345):
    label_map = {}
    
    for (i, label) in enumerate(label_list):
        label_map[int(label)] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        label=label_map[int(example.label)]
        
        encoded_dict = tokenizer.encode_plus(
            example.text_a,  
            add_special_tokens=True,  
            max_length=256,  
            pad_to_max_length=True,
            return_tensors='pt',  
        )

        features.append(
            InputFeatures(input_ids=encoded_dict['input_ids'],
                          token_type_ids=encoded_dict['token_type_ids'],
                          attention_mask=encoded_dict['attention_mask'],
                          lables=label
                          ))
    return features

def convert_examples_to_features_error(examples, label_list, max_seq_length, tokenizer, seed=12345):
    label_map = {}
    
    for (i, label) in enumerate(label_list):
        label_map[int(label)] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        label=label_map[int(example.label)]
        
        encoded_dict = tokenizer.encode_plus(
            example.text_a,  
            add_special_tokens=True,  
            max_length=256,  
            pad_to_max_length=True,
            return_tensors='pt',  
        )

        features.append(
            errorFeatures(input_ids=encoded_dict['input_ids'],
                          token_type_ids=encoded_dict['token_type_ids'],
                          attention_mask=encoded_dict['attention_mask'],
                          lables=label,
                          confidence = example.confidence
                          ))
    return features

def convert_examples_to_features_dev(examples, label_list, max_seq_length, tokenizer, seed=12345):
    features = []
    for (ex_index, example) in enumerate(examples):
        temp = []
        x=example.label
        a = x.strip('[]')
        b = a.split(',')
        for j in b:
            temp.append(float(j))

        label = temp

        encoded_dict = tokenizer.encode_plus(
            example.text_a,  
            add_special_tokens=True,  
            max_length=256,  
            pad_to_max_length=True,
            return_tensors='pt',  
        )

        features.append(
            InputFeatures(input_ids=encoded_dict['input_ids'],
                          token_type_ids=encoded_dict['token_type_ids'],
                          attention_mask=encoded_dict['attention_mask'],
                          lables=label
                          ))
    return features

def prepare_data_classfiction_c(features):
    all_input_ids,all_token_type_ids,all_attention_mask,all_labels,all_confidences=[],[],[],[],[]
    for f in features:
        all_input_ids.append(f.input_ids)
        all_token_type_ids.append(f.token_type_ids)
        all_attention_mask.append(f.attention_mask)
        all_labels.append(f.lables)
        all_confidences.append(f.confidence)

    all_input_ids=torch.cat(all_input_ids,dim=0)
    all_token_type_ids=torch.cat(all_token_type_ids,dim=0)
    all_attention_mask=torch.cat(all_attention_mask,dim=0)


    all_input_ids = torch.LongTensor(all_input_ids)
    all_token_type_ids = torch.LongTensor(all_token_type_ids)
    all_attention_mask = torch.LongTensor(all_attention_mask)
    all_labels = torch.LongTensor(all_labels)
    all_confidences = torch.LongTensor(all_confidences)

    tensor_data = Data.TensorDataset( all_input_ids, all_token_type_ids, all_attention_mask,
                                all_labels,all_confidences)
    return tensor_data

def prepare_data_classfiction(features):
    all_input_ids,all_token_type_ids,all_attention_mask,all_labels,all_confidences=[],[],[],[],[]
    for f in features:
        all_input_ids.append(f.input_ids)
        all_token_type_ids.append(f.token_type_ids)
        all_attention_mask.append(f.attention_mask)
        all_labels.append(f.lables)


    all_input_ids=torch.cat(all_input_ids,dim=0)
    all_token_type_ids=torch.cat(all_token_type_ids,dim=0)
    all_attention_mask=torch.cat(all_attention_mask,dim=0)


    all_input_ids = torch.LongTensor(all_input_ids)
    all_token_type_ids = torch.LongTensor(all_token_type_ids)
    all_attention_mask = torch.LongTensor(all_attention_mask)
    all_labels = torch.LongTensor(all_labels)

    tensor_data = Data.TensorDataset( all_input_ids, all_token_type_ids, all_attention_mask,
                                all_labels)
    return tensor_data

def compute_dev_loss(model, dev_dataloader,label_list,crition):
    model.eval()  
    sum_loss = 0.
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        label=batch[3]
        logits = model(batch[0], batch[2], batch[1],flag_embding)
        loss = crition_ce(logits, label)
        sum_loss += loss.item()
    return sum_loss


def evaluate(model, data_loader,args):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, batch in (enumerate(data_loader)):
            batch = tuple(t.to(device) for t in batch)
            y=batch[3]
            y_pred= model(batch[0], batch[2], batch[1],flag_embding)
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())

    
    precision = precision_score(val_true, val_pred,average='micro')
    accuracy = accuracy_score(val_true, val_pred)
    recall = recall_score(val_true, val_pred,average='micro')
    val = {
        'precision_score':precision,
        'accuracy_score':accuracy,
        'recall_score':recall
    }

    json_str = json.dumps(val)
    with open(os.path.join(args.output_dir,'evalResult.json'), 'w') as json_file:
        json_file.write(json_str)

    print('evaluate',val)

def train_eval(args):
    task_name = args.task_name
    
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    
    processor = get_task_processor(task_name, args.data_dir)
    label_list = processor.get_labels(task_name)
    num_classes = len(label_list)

    
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()
    error_examples = processor.get_error_examples()
    erroraug_examples = processor.get_erroraug_examples()

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese/')
    
    model = MyModel('bert-base-chinese/', 768, num_classes)
    model1 = MyModel('robert_model/', 768, num_classes)
    model.to(device)
    model1.to(device)

    
    print('load train data')
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  args.max_seq_length,
                                                  tokenizer, args.seed)
    train_data = prepare_data_classfiction(train_features)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    
    print('load error data')
    error_features = convert_examples_to_features_error(error_examples, label_list,
                                                  args.max_seq_length,
                                                  tokenizer, args.seed)
    error_data = prepare_data_classfiction_c(error_features)
    error_sampler = SequentialSampler(error_data)
    error_dataloader = DataLoader(error_data, sampler=error_sampler,
                                  batch_size=args.train_batch_size)
    error_iter = iter(error_dataloader)

    
    print('load aug data')
    erroraug_features = convert_examples_to_features_error(erroraug_examples, label_list,
                                                  args.max_seq_length,
                                                  tokenizer, args.seed)
    erroraug_data = prepare_data_classfiction_c(erroraug_features)
    erroraug_sampler = SequentialSampler(erroraug_data)
    erroraug_dataloader = DataLoader(erroraug_data, sampler=erroraug_sampler,
                                  batch_size=args.train_batch_size)
    erroraug_iter = iter(erroraug_dataloader)

    
    print('load dev data')
    dev_features = convert_examples_to_features(dev_examples, label_list,
                                                args.max_seq_length,
                                                tokenizer, args.seed)
    dev_data = prepare_data_classfiction(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=int(args.train_batch_size))

    num_train_steps = int(len(train_features) / args.train_batch_size * args.num_train_epochs)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    best_dev_loss = float('inf')

    
    print('------training bert------')
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        avg_loss = 0.
        model.train()
        model1.eval()
        for step,batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            label=batch[3]

            logits = model(batch[0],batch[2],batch[1],flag_embding)
            
            
            loss = crition_ce(logits,label)
            
            optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            avg_loss = 0.
            
        dev_loss = compute_dev_loss(model, dev_dataloader, label_list,crition_ce)

        print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print("Saving model. Best dev so far {}".format(best_dev_loss))
            save_bert_path = os.path.join(args.output_dir, 'first_bert.pt')
            torch.save(model.state_dict(), save_bert_path)

    print('------training robert------')
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        avg_loss = 0.
        model1.train()
        model.eval()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            label = batch[3]

            logits = model(batch[0], batch[2], batch[1], flag_embding)
            
            
            loss = crition_ce(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            avg_loss = 0.
            
        dev_loss = compute_dev_loss(model1, dev_dataloader, label_list, crition_ce)

        print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print("Saving model. Best dev so far {}".format(best_dev_loss))
            save_robert_path = os.path.join(args.output_dir, 'first_robert.pt')
            torch.save(model.state_dict(), save_robert_path)

    
    if os.path.exists(save_bert_path):
        model.load_state_dict(torch.load(save_bert_path), strict=False)
    else:
        raise ValueError("Unable to find the saved model at {}".format(save_bert_path))

    if os.path.exists(save_robert_path):
        model1.load_state_dict(torch.load(save_robert_path), strict=False)
    else:
        raise ValueError("Unable to find the saved model at {}".format(save_robert_path))

    
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        avg_loss = 0.
        model.train()
        model1.eval()  
        for step, batch_x in enumerate(train_dataloader):
            
            batch_x = tuple(t.to(device) for t in batch_x)
            label_x = batch_x[3]
            label_x = one_hot(label_x, num_classes)

            
            batch_u = error_iter.next()
            batch_u = tuple(t.to(device) for t in batch_u)
            label_u = batch_u[3]
            label_u = one_hot(label_u , num_classes)
            confidence = batch_u[4]

            
            batch_u1 = erroraug_iter.next()
            batch_u1 = tuple(t.to(device) for t in batch_u1)

            logits_u1 = model1(batch_u1[0], batch_u1[2], batch_u1[1], flag_embding)
            log_prob_u1 = torch.nn.functional.log_softmax(logits_u1, dim=1)

            relabel = confidence * label_u + (1-confidence) * log_prob_u1  
            relabel = relabel ** (1 / args.T)  

            relabel = relabel / relabel.sum(dim=1, keepdim=True)  
            relabel = relabel.detach()

            
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            l.to(device)

            all_input_ids = torch.cat([batch_x[0], batch_u[0]], dim=0)
            all_token_type_ids = torch.cat([batch_x[1], batch_u[1]], dim=0)
            all_attention_mask = torch.cat([batch_x[2], batch_u[2]], dim=0)
            all_targets = torch.cat([label_x,relabel], dim=0)
            idx = torch.randperm(all_input_ids.size(0))
            idx.to(device)

            
            input_ids_a, input_ids_b = all_input_ids, all_input_ids[idx]
            token_type_ids_a,token_type_ids_b = all_token_type_ids, all_token_type_ids[idx]
            attention_mask_a,attention_mask_b = all_attention_mask, all_attention_mask[idx]
            label_a, label_b = all_targets, all_targets[idx]

            mix_input_ids = l * input_ids_a + (1 - l) * input_ids_b
            mix_token_type_ids = l * token_type_ids_a + (1 - l) * token_type_ids_b
            mix_attention_mask = l * attention_mask_a + (1 - l) * attention_mask_b
            mix_label = l * label_a + (1 - l) * label_b

            logits = model(mix_input_ids, mix_attention_mask, mix_token_type_ids, flag_embding)
            logits_x = logits[:args.train_batch_size]
            logits_u = logits[args.train_batch_size:]

            log_prob_x = torch.nn.functional.log_softmax(logits_x, dim=1)
            

            loss_x = -torch.sum(log_prob_x  * mix_label) / (args.train_batch_size)  
            loss_u = crition_mse(logits_u, mix_label)  
            lamb.to(device)
            loss = loss_x + lamb * loss_u

            optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            avg_loss = 0.
            
        dev_loss = compute_dev_loss(model, dev_dataloader, label_list, crition_ce)

        print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print("Saving model. Best dev so far {}".format(best_dev_loss))
            best_bert_path = os.path.join(args.output_dir, 'best_bert.pt')
            torch.save(model.state_dict(), best_bert_path)

    if os.path.exists(best_bert_path):
        model.load_state_dict(torch.load(best_bert_path), strict=False)
    else:
        raise ValueError("Unable to find the saved model at {}".format(best_bert_path))

    test_examples = processor.get_test_examples()
    test_features = convert_examples_to_features(test_examples, label_list,
                                                      args.max_seq_length,
                                                      tokenizer, args.seed)
    test_data = prepare_data_classfiction(test_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler,
                                     batch_size=int(args.train_batch_size))
    evaluate(model, test_dataloader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--data_dir", default="CTCC/", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="CTCC/nopretrain/", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--task_name", default="dianxin", type=str,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    
    
    parser.add_argument('--cache', default="transformers_cache", type=str)

    parser.add_argument("--train_batch_size", default=12, type=int,
                            help="Total batch size for training.")
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=4e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--sample_num', type=int, default=1,
                        help="sample number")
    parser.add_argument('--sample_ratio', type=int, default=7,
                        help="sample ratio")
    
    
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')

    args = parser.parse_args()

    print(args)
    
    train_eval(args)











