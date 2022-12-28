from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
import logging
from tqdm import tqdm
from datetime import datetime
import pytz
import argparse
from torch.autograd import Variable
from loader import *
from sklearn.metrics import roc_auc_score
from icecream import ic
from model_backup import GAT, SpGAT

# torch.set_printoptions(threshold=np.inf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='path to save results and logger files.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu used to train.')
    parser.add_argument('--seed', type=int, default=86,
                        help='Random seed.')

    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu.')

    parser.add_argument('--nheads', type=int, default=3,
                        help='Number of head attentions.')

    parser.add_argument('--feature_path', type=str, default='0',
                        help='Path of feature')
    parser.add_argument('--exp_train_path', type=str, default='0',
                        help='Path of cross validation split file')
    parser.add_argument('--exp_test_path', type=str, default=None,
                        help='Path of cross validation split file')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Path of cross validation split file')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Epochs to train')

    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate for Adam')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay for Adam')

    parser.add_argument('--hid_dim', type=int, default=160,
                        help='Dimension of hidden feature')
    parser.add_argument('--node_num', type=int, default=2,
                        help='Number of nodes')

    parser.add_argument('--layers', type=int, default=1,
                        help='Number of attention layers.')
    parser.add_argument('--dropout_rate', type=float, default=0,
                        help='Dropout rate in AGNN model')
    parser.add_argument('--class_type', type=str, default='AD_NC',
                        help='2-class modify')
    parser.add_argument('--dataset', type=str, default='mean_bold',
                        help='choices: [hidden,avg] | avg datasets are for contrastive study')

    args = parser.parse_args()

    """ Use CUDA and Random Seed """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    """ create folders for results to save """
    timenow = datetime.strftime(datetime.now(pytz.timezone('Asia/Shanghai')), '%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    config_name = 'C' + str(args.node_num) + '_H' + str(args.hid_dim) \
                  + '_L' + str(args.layers) + '_d' + str(int(args.dropout_rate * 100)) \
                  + '_E' + str(args.epochs)

    logdir = os.path.join(args.save_dir, config_name, args.dataset, args.class_type, 'logs', timenow)
    savedir = os.path.join(args.save_dir, config_name, args.dataset, args.class_type, 'checkpoints', timenow)
    shotdir = os.path.join(args.save_dir, config_name, args.dataset, args.class_type, 'snapshot', timenow)
    perfdir = os.path.join(args.save_dir, config_name, args.dataset, args.class_type, 'test_performance', timenow)
    train_info_dir = os.path.join(args.save_dir, config_name, args.dataset, args.class_type, 'train_performance', timenow)

    os.makedirs(logdir, exist_ok=False)
    os.makedirs(savedir, exist_ok=False)
    os.makedirs(shotdir, exist_ok=False)
    os.makedirs(perfdir, exist_ok=False)
    os.makedirs(train_info_dir, exist_ok=False)

    print(
        'Results save at: {}.\nLog save at: {}.\nCheckpoints save at {}.\nSnapshot save at {}.\nPerformance save at: {}\n'
        .format(args.save_dir, logdir, savedir, shotdir, perfdir))

    # There wasn't epoch_test_loss in 'test_performance.csv' originally.
    with open(perfdir + '/test_performance.csv', 'w') as csv_file:
        test_writer = csv.writer(csv_file)
        test_writer.writerow(
            ['epoch', 'TP', 'TN', 'FP', 'FN', 'p', 'r', 'TPR', 'FPR', 'F1', 'acc', 'auc_output', 'auc_pred', 'loss', 'epoch_test_loss'])

    # There wasn't epoch_train_loss in 'train_performance.csv' originally.
    with open(train_info_dir + '/train_performance.csv', 'w') as csv_file:
        test_writer = csv.writer(csv_file)
        test_writer.writerow(
            ['epoch', 'TP', 'TN', 'FP', 'FN', 'p', 'r', 'TPR', 'FPR', 'F1', 'acc', 'auc_output', 'auc_pred', 'loss', 'epoch_train_loss'])
    writer = SummaryWriter(logdir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(shotdir + '/' + 'snapshot.log', encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.info(str(args))

    """ Load dataset """

    args.feature_path = '/Users/fengyiting/Desktop/UCAS/dataset/mean_bold_data'
    # args.feature_path = '/Users/fengyiting/Desktop/UCAS/dataset/Sub_Dataset/Sub_Dataset'
    args.exp_train_path = '/Users/fengyiting/Desktop/UCAS/split_for_test/trainset_0.csv'
    args.exp_test_path = '/Users/fengyiting/Desktop/UCAS/split_for_test/testset_0.csv'
    # args.exp_train_path = '/Users/fengyiting/Desktop/UCAS/dataset/fold_split_AD_NC/trainset_0.csv'
    # args.exp_test_path = '/Users/fengyiting/Desktop/UCAS/dataset/fold_split_AD_NC/testset_0.csv'

    if args.dataset == 'hidden':
        trainset = fMRI_Hidden_Dataset(args.feature_path, args.exp_train_path, args.class_type)
        testset = fMRI_Hidden_Dataset(args.feature_path, args.exp_test_path, args.class_type)
    elif args.dataset in ['ALFF_REHO', 'ALFF', 'REHO']:
        trainset = fMRI_REHO_ALFF_Dataset(feature_dir=args.feature_path, exp_path=args.exp_train_path,
                                          cishu=args.resnet_feature_cishu, phase=args.dataset, mode=args.class_type,
                                          state='train')
        testset = fMRI_REHO_ALFF_Dataset(feature_dir=args.feature_path, exp_path=args.exp_test_path,
                                         cishu=args.resnet_feature_cishu, phase=args.dataset, mode=args.class_type,
                                         state='test')
    else:
        trainset = mean_bold_Dataset(args.feature_path, args.exp_train_path, args.class_type)
        testset = mean_bold_Dataset(args.feature_path, args.exp_test_path, args.class_type)

    print('\nSize of training set: {}.\nSize of validation set: {}.\n'.format(len(trainset), len(testset)))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print('{} iterations per expoch for training.'.format(len(trainloader)))
    print('{} iterations per expoch for test batch.'.format(len(testloader)))

    # build model
    feature_dim = testset[0]['feature'].shape[1]
    model = GAT(nfeat=feature_dim,
                nhid=args.hid_dim,
                nclass=args.node_num,
                dropout=args.dropout_rate,
                nheads=args.nheads,
                alpha=args.alpha)

    model.train()
    CE_loss = torch.nn.CrossEntropyLoss()
    best_performance = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    """ Training """
    for epoch in range(args.epochs):

        print("Epoch %d / %d : " % (epoch + 1, args.epochs))
        epoch_str = epoch+1
        epoch_str = str(epoch_str)

        """
        if isinstance(optimizer, optim.SGD) and args.lr_mode == "cosine":
            lr = adjust_learning_rate(optimizer, args.epochs - args.warmup_epochs, args.lr,
                                      args.lr_rampdown_epochs, args.eta_min)  # args.lr
           
            for param_group in optimizer.param_groups:
                print('lr --->', lr)
                param_group['lr'] = lr
        """
        loss_epoch = 0
        correct = 0
        total_num = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        label_all, pred_all, out_all = [], [], []

        for i_batch, sampled_batch in enumerate(trainloader):
            sub_batch, adj_batch, label_batch = sampled_batch['feature'], sampled_batch['adj'], sampled_batch['label']
            sub_batch, adj_batch, label_batch = Variable(sub_batch), Variable(adj_batch), Variable(
                label_batch)

            outputs = model(sub_batch, adj_batch)

            loss = CE_loss(outputs, label_batch)
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict = torch.argmax(outputs, dim=1)
            correct += (predict == label_batch).sum()
            total_num += len(label_batch)

            TP += ((predict == 1) & (label_batch == 1)).cpu().sum()
            TN += ((predict == 0) & (label_batch == 0)).cpu().sum()
            FN += ((predict == 0) & (label_batch == 1)).cpu().sum()
            FP += ((predict == 1) & (label_batch == 0)).cpu().sum()

            out_all.extend(outputs[:, 1].detach().cpu().numpy())
            pred_all.extend(predict.detach().cpu().numpy())
            label_all.extend(label_batch.cpu().numpy())

            logging.info('iteration %d : loss : %f accuarcy: %f%% '
                         % (i_batch + 1, loss.item(), float(correct * 100) / total_num))

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        auc1 = roc_auc_score(label_all, out_all)
        auc2 = roc_auc_score(label_all, pred_all)

        logging.info(
            'TP %d : TN : %d FN: %d FP: %d p: %f%% r: %f%% \nTPR: %f%% FPR: %f%% F1: %f%% acc: %f%% auc_output: %f%% auc_pred: %f%%'
            % (TP, TN, FN, FP, p * 100, r * 100, TPR * 100, FPR * 100, F1 * 100, acc * 100, auc1 * 100, auc2 * 100))

        epoch_loss = loss_epoch / len(trainloader)
        # total loss/acc written in tensorbox
        writer.add_scalar('loss/train_loss', epoch_loss, epoch + 1)
        writer.add_scalar('acc/train_acc', float(correct * 100) / total_num, epoch + 1)

        # print total loss/acc
        logging.info('epoch %d : epoch_loss : %f acc: %f%%' % (
            epoch + 1, epoch_loss, float(correct * 100) / (float(args.batch_size) * (i_batch + 1))))

        # add 'epoch_loss' in train_performance.csv
        with open(train_info_dir + '/train_performance.csv', 'a+') as csv_file:
            test_writer = csv.writer(csv_file)
            test_writer.writerow([str(epoch + 1), str(TP.item()), str(TN.item()),
                                  str(FP.item()), str(FN.item()), str(p.item()), str(r.item()),
                                  str(TPR.item()), str(FPR.item()), str(F1.item()), str(acc.item()),
                                  str(auc1), str(auc2), str(loss.item()), str(epoch_loss)])

        """ test """
        model.eval()
        if epoch % 10 == 0:
            loss_epoch_wo_dropout = 0
            correct = 0
            total_num = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            label_all, pred_all, out_all = [], [], []
            for i_batch, sampled_batch in enumerate(trainloader):
                sub_batch, adj_batch, label_batch = sampled_batch['feature'], sampled_batch['adj'], sampled_batch[
                    'label']
                sub_batch, adj_batch, label_batch = Variable(sub_batch), Variable(adj_batch), Variable(
                    label_batch)

                outputs = model(sub_batch, adj_batch)

                loss = CE_loss(outputs, label_batch)
                loss_epoch_wo_dropout += loss.item()

                predict = torch.argmax(outputs, dim=1)
                correct += (predict == label_batch).sum()
                total_num += len(label_batch)

                TP += ((predict == 1) & (label_batch == 1)).cpu().sum()
                TN += ((predict == 0) & (label_batch == 0)).cpu().sum()
                FN += ((predict == 0) & (label_batch == 1)).cpu().sum()
                FP += ((predict == 1) & (label_batch == 0)).cpu().sum()

                out_all.extend(outputs[:, 1].detach().cpu().numpy())
                pred_all.extend(predict.detach().cpu().numpy())
                label_all.extend(label_batch.cpu().numpy())

                logging.info('Trainin wo dropout iteration %d : loss : %f accuarcy: %f%% '
                             % (i_batch + 1, loss.item(), float(correct * 100) / total_num))

            p = TP / (TP + FP)
            r = TP / (TP + FN)
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            auc1 = roc_auc_score(label_all, out_all)
            auc2 = roc_auc_score(label_all, pred_all)
            epoch_loss_wo_dropout = loss_epoch_wo_dropout / len(trainloader)
            writer.add_scalar('loss/train_loss_wo_dropout', epoch_loss_wo_dropout, epoch)
            writer.add_scalar('acc/train_acc_wo_dropout', float(correct * 100) / total_num, epoch)

        loss_epoch_test = 0
        correct_test = 0
        total_test_num = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        pred_all_test = []
        out_all_test = []
        label_all_test = []

        #
        for i_batch, sampled_batch in enumerate(tqdm(testloader)):
            sub_batch, adj_batch, label_batch = sampled_batch['feature'], sampled_batch['adj'], sampled_batch['label']
            sub_batch, adj_batch, label_batch = Variable(sub_batch), Variable(adj_batch), Variable(label_batch)

            batch_index = i_batch + 1
            batch_index = str(batch_index)

            outputs = model(sub_batch, adj_batch)

            # when epoch is over xx, it begins to save attention matrix
            if epoch > 9000:
                attendir = os.path.join(args.save_dir, config_name, args.dataset, args.class_type, 'attention', timenow, epoch_str)
                os.makedirs(attendir, exist_ok=True)

                attention1 = attention1.detach().numpy()
                attention2 = attention2.detach().numpy()
                attention3 = attention3.detach().numpy()
                attention4 = attention4.detach().numpy()
                attention5 = attention5.detach().numpy()
                attention6 = attention6.detach().numpy()

                for batch_in, slice_2d in enumerate(attention1):
                    batch_in = str(batch_in + 1)
                    dir = os.path.join(attendir + '/' + 'layer_1' + '/')
                    os.makedirs(dir, exist_ok=True)
                    filename = dir + batch_index + '_' + batch_in + '.txt'
                    np.savetxt(X=slice_2d, fname=filename)
                for batch_in, slice_2d in enumerate(attention2):
                    batch_in = str(batch_in)
                    dir = os.path.join(attendir + '/' + 'layer_2' + '/')
                    os.makedirs(dir, exist_ok=True)
                    filename = dir + batch_index + '_' + batch_in + '.txt'
                    np.savetxt(X=slice_2d, fname=filename)
                for batch_in, slice_2d in enumerate(attention3):
                    batch_in = str(batch_in)
                    dir = os.path.join(attendir + '/' + 'layer_3' + '/')
                    os.makedirs(dir, exist_ok=True)
                    filename = dir + batch_index + '_' + batch_in + '.txt'
                    np.savetxt(X=slice_2d, fname=filename)
                for batch_in, slice_2d in enumerate(attention4):
                    batch_in = str(batch_in)
                    dir = os.path.join(attendir + '/' + 'layer_4' + '/')
                    os.makedirs(dir, exist_ok=True)
                    filename = dir + batch_index + '_' + batch_in + '.txt'
                    np.savetxt(X=slice_2d, fname=filename)
                for batch_in, slice_2d in enumerate(attention5):
                    batch_in = str(batch_in)
                    dir = os.path.join(attendir + '/' + 'layer_5' + '/')
                    os.makedirs(dir, exist_ok=True)
                    filename = dir + batch_index + '_' + batch_in + '.txt'
                    np.savetxt(X=slice_2d, fname=filename)
                for batch_in, slice_2d in enumerate(attention6):
                    batch_in = str(batch_in)
                    dir = os.path.join(attendir + '/' + 'layer_6' + '/')
                    os.makedirs(dir, exist_ok=True)
                    filename = dir + batch_index + '_' + batch_in + '.txt'
                    np.savetxt(X=slice_2d, fname=filename)


            loss = CE_loss(outputs, label_batch)
            loss_epoch_test += loss.item()

            predict = torch.argmax(outputs, dim=1)
            correct_test += (predict == label_batch).sum()
            total_test_num += len(label_batch)

            TP += ((predict == 1) & (label_batch == 1)).cpu().sum()
            TN += ((predict == 0) & (label_batch == 0)).cpu().sum()
            FN += ((predict == 0) & (label_batch == 1)).cpu().sum()
            FP += ((predict == 1) & (label_batch == 0)).cpu().sum()

            out_all_test.extend(outputs[:, 1].detach().cpu().numpy())
            pred_all_test.extend(predict.detach().cpu().numpy())
            label_all_test.extend(label_batch.cpu().numpy())

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        auc1 = roc_auc_score(label_all_test, out_all_test)
        auc2 = roc_auc_score(label_all_test, pred_all_test)

        performance = float(correct_test * 100) / total_test_num

        # the whole loss in each epoch
        epoch_loss_test = loss_epoch_test / len(testloader)

        # the 'loss' stands for a batch's loss in each epoch.
        # add 'epoch_loss_test' in logging.info
        logging.info('\nTest iteration %d : loss : %f epoch_loss_test: %f accuarcy: %f%% \n'
                     % (i_batch + 1, loss.item(), epoch_loss_test, float(correct_test * 100) / total_test_num))
        logging.info(
            'TP %d : TN : %d FN: %d FP: %d p: %f%% r: %f%% \nTPR: %f%% FPR: %f%% F1: %f%% acc: %f%% auc output: %f%% auc pred: %f%%'
            % (TP, TN, FN, FP, p * 100, r * 100, TPR * 100, FPR * 100, F1 * 100, acc * 100, auc1 * 100, auc2 * 100))

        # add 'epoch_loss_test' in test_performance.csv
        with open(perfdir + '/test_performance.csv', 'a+') as csv_file:
            test_writer = csv.writer(csv_file)
            test_writer.writerow([str(epoch + 1), str(TP.item()), str(TN.item()),
                                  str(FP.item()), str(FN.item()), str(p.item()), str(r.item()),
                                  str(TPR.item()), str(FPR.item()), str(F1.item()), str(acc.item()),
                                  str(auc1), str(auc2), str(loss.item()), str(epoch_loss_test)])

        model.train()
        # write an epoch's loss in tensorbox
        epoch_loss_test = loss_epoch_test / len(testloader)
        writer.add_scalar('loss/test_loss', epoch_loss_test, epoch + 1)
        writer.add_scalar('acc/test_acc', float(correct_test * 100) / total_test_num, epoch + 1)

        if best_performance < performance:
            best_performance = performance
            torch.save(model, os.path.join(savedir, 'checkpoint_' + str(epoch + 1) + '.pth'))
            logging.info('Saved checkpoint: {}'.format(os.path.join(savedir, 'checkpoint_' + str(epoch + 1) + '.pth')))
            logging.info('Best performance: {:4f}%%'.format(best_performance))

    logging.info('Best performance: {:4f}%%'.format(best_performance))
    writer.close()


def adjust_learning_rate(optimizer, epoch, lr, lr_rampdown_epochs=None, eta_min=None):
    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if lr_rampdown_epochs:
        # assert lr_rampdown_epochs >= epochs
        lr = eta_min + (lr - eta_min) * cosine_rampdown(epoch, lr_rampdown_epochs)

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

    return lr


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


if __name__ == "__main__":
    main()
