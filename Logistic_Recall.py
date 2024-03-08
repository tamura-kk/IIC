import argparse
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from config import get_config
import multiprocessing


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--no', default=0)
#     return parser.parse_args()

#     args = get_args()

def parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):

    no = config.MODEL.NUMBER
    model_name = config.MODEL.TYPE
    data_type = config.DATA.TYPE
    
    base_path = os.path.join('result', str(data_type), str(model_name), str(no), 'test')

    data = pd.read_csv(os.path.join(base_path, 'CountResult_{}.csv'.format(no)))


    result_path = os.path.join(base_path, 'logistic_kfold_re')

    os.makedirs(result_path, exist_ok=True)

    FOLD = 5

    class_num = list(data.columns[3:])
    class_list = [str(num) for num in class_num]

    print(class_list)

    X = data[class_list]
    Y = data['type']

    print('X_data', X.shape)
    print('Y_data', Y.shape)

    print('-'*50)
    print('###k_fold_start###')

    skf = StratifiedKFold(n_splits=FOLD, random_state=0, shuffle=True)

    test_acc_all = []
    test_recall_all = []
    test_precision_all = []
    test_f1_all = []
    test_matrix = []
    all_acc_all = []
    recall_all = []
    f1_all = []
    precision_all = []
    all_matrix = []
    params_csv = []

    result_data = data.drop(class_list, axis=1)
    on_off_data = data.drop(class_list, axis=1)


    for fold_num, (train_index, test_index) in enumerate(skf.split(X, Y)):

        print(fold_num)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        print('train = ', X_train.shape)
        print('test = ', X_test.shape)
        print('-'*50)
        
        lr = LogisticRegression(fit_intercept=True, random_state=0)
        lr.fit(X_train, Y_train)

        there_fold = Y_train.sum() / len(Y_train)

        Y_pred = (lr.predict_proba(X_test)[:, 1] > there_fold).astype(int)

        Y_pred = (lr.predict_proba(X_test)[:, 1] > there_fold).astype(int)
        test_acc_all.append(accuracy_score(y_true=Y_test, y_pred=Y_pred))
        test_recall_all.append(recall_score(y_true=Y_test, y_pred=Y_pred))
        test_precision_all.append(precision_score(y_true=Y_test, y_pred=Y_pred))
        test_f1_all.append(f1_score(y_true=Y_test, y_pred=Y_pred))
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
        test_matrix.append([tp, fn, fp, tn, tp+tn])

        Y_pred = (lr.predict_proba(X)[:, 1] > there_fold).astype(int)
        all_acc_all.append(accuracy_score(y_true=Y, y_pred=Y_pred))
        recall_all.append(recall_score(y_true=Y, y_pred=Y_pred))
        precision_all.append(precision_score(y_true=Y, y_pred=Y_pred))
        f1_all.append(f1_score(y_true=Y, y_pred=Y_pred))
        tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()
        all_matrix.append([tp, fn, fp, tn, tp+tn])

        coef = pd.DataFrame(lr.coef_.reshape((-1, 1)),
                            class_list, columns=['regression coefficient'])
        interce = pd.DataFrame([lr.intercept_], index=['constant term'], columns=['regression coefficient'])
        there_fold = pd.DataFrame([there_fold], index=['th'], columns=['regression coefficient'])
        df_coef = pd.concat([coef, interce, there_fold])
        params_csv.append(df_coef)

        predict = lr.predict_proba(X)
        predict = predict[:, 1]
        result_data['pred' + str(fold_num)] = predict
        on_off_data['pred_int' + str(fold_num)] = Y_pred


    #save accuracy
    test_acc_all.append(np.mean(np.array(test_acc_all)))
    all_acc_all.append(np.mean(np.array(all_acc_all)))    # 平均値を追加
    label = [str(i) for i in range(FOLD)]
    label.append("average")
    label = pd.DataFrame(label, columns=['Kfold'])
    test_acc_all = pd.DataFrame(test_acc_all, columns=['test_accuracy'])
    all_acc_all = pd.DataFrame(all_acc_all, columns=['all_data_accuracy'])
    save_acc_csv = pd.concat([label, test_acc_all, all_acc_all], axis=1)
    save_acc_csv.to_csv(os.path.join(result_path,'acc_{}.csv'.format(no)))

    #save recall
    test_recall_all.append(np.mean(np.array(test_recall_all)))
    recall_all.append(np.mean(np.array(recall_all)))    # 平均値を追加
    test_recall_all = pd.DataFrame(test_recall_all, columns=['test_recall'])
    recall_all = pd.DataFrame(recall_all, columns=['alldata_recall'])
    save_recall_csv = pd.concat([label, test_recall_all, recall_all], axis=1)
    save_recall_csv.to_csv(os.path.join(result_path, 'recall_{}.csv'.format(no)))

    #save precision
    test_precision_all.append(np.mean(np.array(test_precision_all)))
    precision_all.append(np.mean(np.array(precision_all)))    # 平均値を追加
    test_precision_all = pd.DataFrame(
        test_precision_all, columns=['test_precision'])
    precision_all = pd.DataFrame(precision_all, columns=['alldata_precision'])
    save_precision_csv = pd.concat(
        [label, test_precision_all, precision_all], axis=1)
    save_precision_csv.to_csv(os.path.join(result_path, 'precision_{}.csv'.format(no)))

    #save F1_score
    test_f1_all.append(np.mean(np.array(test_f1_all)))
    f1_all.append(np.mean(np.array(f1_all)))    # 平均値を追加
    test_f1_all = pd.DataFrame(test_f1_all, columns=['test_f1'])
    f1_all = pd.DataFrame(f1_all, columns=['alldata_f1'])
    save_f1_csv = pd.concat([test_f1_all, f1_all], axis=1)
    save_f1_csv.to_csv(os.path.join(result_path, 'f1_{}.csv'.format(no)))

    #save acc, recall, precision, f1_score
    arpf = pd.concat([label, test_acc_all, test_recall_all,
                    test_precision_all, test_f1_all], axis=1)
    arpf.to_csv(os.path.join(result_path, 'arpf_{}.csv'.format(no)))

    #save params
    print(len(params_csv))
    params_csv = np.array(params_csv)  # [5,14,1]
    params_csv = np.squeeze(params_csv)
    params_csv = np.transpose(params_csv, (1, 0))

    #print(params_csv.shape)
    class_list.append("定数項")
    class_list.append("閾値")
    params_csv = pd.DataFrame(params_csv, index=class_list)
    print(params_csv)
    params_csv.to_csv(os.path.join(result_path, 'parameter_{}.csv'.format(no)))

    #save predict
    result_data.to_csv(os.path.join(result_path, 'result_{}.csv'.format(no)), index=False)
    on_off_data.to_csv(os.path.join(result_path, 'result_int_{}.csv'.format(no)), index=False)

    print("-"*100)
    print(test_acc_all, '\tmean=', np.mean(test_acc_all))
    print(all_acc_all, '\tmean=', np.mean(all_acc_all))
    print("-"*100)

    print("confusion_matrix")
    test_matrix = pd.DataFrame(test_matrix, columns=[
                            'tp', 'fn', 'fp', 'tn', 'tp+tn'])
    all_matrix = pd.DataFrame(
        all_matrix, columns=['tp', 'fn', 'fp', 'tn', 'tp+tn'])
    test_matrix.to_csv(os.path.join(result_path, 'test_matrix_{}.csv'.format(no)))
    all_matrix.to_csv(os.path.join(result_path, 'all_matrix_{}.csv'.format(no)))
    print(test_matrix)
    print("-"*100)
    print(all_matrix)
    print("-"*100)

if __name__ == '__main__':
    args, config = parse_option()
    multiprocessing.set_start_method('spawn')
    main(config)
