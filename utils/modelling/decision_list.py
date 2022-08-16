from interpret.glassbox import DecisionListClassifier
import pickle
from sklearn.model_selection import train_test_split
from utils.checks import *
from utils.modelling.performance import *

def trainDecisionList(X_train, y_train, params, model_type, logging):
    # if 'feature_names' not in params.keys():
    #     params['feature_names'] = X_train.columns
    if model_type == 'classification':
        clf = DecisionListClassifier(**params)
    else:
        print('Only classification available')
        logging.warning('Only classification available')

    clf.fit(X_train, y_train)
    print('Trained decision list \n')
    logging.info('Trained decision list')

    return clf

def plotTreeStructureSave(clf, given_name):

    plt.figure(figsize=(30,30))

    tree.plot_tree(clf, fontsize=10, feature_names=clf.feature_names_in_, class_names=clf.classes_)
    plt.savefig('{given_name}/tree_plot.png'.format(given_name=given_name))

    print('Tree structure plot saved')


def featureImportanceSave(clf, given_name):
    importance_df = pd.DataFrame({'importance':clf.feature_importances_, 'feature':clf.feature_names_in_}).sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df.to_csv('{given_name}/feature_importance.csv'.format(given_name=given_name), index=False)

    print('Feature importance csv saved')


def make_model(given_name, datasets, model_type, model_params, post_params, logging):
    # unpack datasets
    X_train = datasets['cv_train']['X']
    y_train = datasets['cv_train']['y']
    X_test = datasets['cv_test']['X']
    y_test = datasets['cv_test']['y']
    X_all = datasets['final_train']['X']
    y_all = datasets['final_train']['y']


    # check if X is a list (CV should be applied in that case)
    if isinstance(X_train, list):
        y_test_pred_list = list()
        y_test_prob_list = list()
        y_test_list = y_test

        for fold_id in range(len(X_train)):
            print('Train model on test data')
            logging.info('Train model on test data')

            # Check if model needs to be calibrated
            if post_params['calibration'] != 'false':
                X_slice_train, X_cal, y_slice_train, y_cal = train_test_split(X_train[fold_id],y_train[fold_id], test_size=0.2,random_state=random.randint(0,100))
            else:
                X_slice_train, y_slice_train = X_train[fold_id], y_train[fold_id]

            clf = trainDecisionList(X_slice_train, y_slice_train, model_params, model_type, logging)
            # logging.info(f'Model params:\n {clf.get_params}')

            if post_params['calibration'] != 'false':
                clf = calibrateModel(clf, X_cal, y_cal, logging, method=post_params['calibration'], final_model=False)

            # discrete predictions
            y_test_pred_list.append(clf.predict(X_test[fold_id]))

            if model_type == 'classification':
                # probability predictions
                y_test_prob_list.append(clf.predict_proba(X_test[fold_id])[:,1])

        # Merge list of prediction lists into one list
        y_test_pred = np.concatenate(y_test_pred_list, axis=0)

        if model_type == 'classification':
            # Merge list of prediction probabilities lists into one list
            y_test_prob = np.concatenate(y_test_prob_list, axis=0)

        y_test = np.concatenate(y_test, axis=0)

    # If just regular train/test split has been applied
    else:
        clf = trainDecisionList(X_train, y_train, model_params, model_type, logging)

        # discrete prediction
        y_test_pred = clf.predict(X_test)

        if model_type == 'classification':
            # probability prediction
            y_test_prob = clf.predict_proba(X_test)[:,1]

    # train model one last time on all samples (upsampled)
    print('Train final model on all data')
    logging.info('Train final model on all data')

    # Check if model needs to be calibrated
    if post_params['calibration'] != 'false':
        X_all, X_cal, y_all, y_cal = train_test_split(X_all, y_all, test_size=0.2, random_state=123)

    clf = trainDecisionList(X_all, y_all, model_params, model_type, logging)

    # Save model in pickled format
    filename = given_name + '/model/decision_tree_{model_type}.sav'.format(model_type=model_type)
    pickle.dump(clf, open(filename, 'wb'))

    # train set prediction of final model
    y_all_pred = clf.predict(X_all)
    if model_type == 'classification':
        y_all_prob = clf.predict_proba(X_all)[:,1]

    if post_params['calibration'] != 'false':
        cal_clf, cal_reg = calibrateModel(clf, X_cal, y_cal, logging, method=post_params['calibration'], final_model=True)

        # Save model in pickled format
        filename = given_name + '/model/decision_tree_calibrated_{model_type}.sav'.format(model_type=model_type)
        pickle.dump(cal_clf, open(filename, 'wb'))

        # train set prediction of final model
        y_all_pred = cal_clf.predict(X_all)

        if model_type == 'classification':
            y_all_prob = cal_clf.predict_proba(X_all)[:, 1]



    if model_type == 'classification':
        # Threshold dependant
        plotConfusionMatrixSave(given_name, y_all, y_all_pred, data_type='final_train')
        plotConfusionMatrixSave(given_name, y_test, y_test_pred, data_type='test')
        classificationReportSave(given_name, y_all, y_all_pred, data_type='final_train')
        classificationReportSave(given_name, y_test, y_test_pred, data_type='test')

        if len(clf.classes_) == 2:
            # Also create pr curve for class 0
            y_all_neg = np.array([1 - j for j in list(y_all)])
            y_all_prob_neg = np.array([1 - j for j in list(y_all_prob)])

            y_test_list_neg = [[1 - j for j in i] for i in y_test_list]
            y_test_prob_list_neg = [[1 - j for j in i] for i in y_test_prob_list]

            # Threshold independant
            plotClassificationCurve(given_name, y_all, y_all_prob, curve_type='roc', data_type='final_train')
            plotClassificationCurve(given_name, y_test_list, y_test_prob_list, curve_type='roc', data_type='test')

            plotClassificationCurve(given_name, y_all, y_all_prob, curve_type='pr', data_type='final_train_class1')
            plotClassificationCurve(given_name, y_all_neg, y_all_prob_neg, curve_type='pr', data_type='final_train_class0')

            plotClassificationCurve(given_name, y_test_list, y_test_prob_list, curve_type='pr', data_type='test_data_class1')
            plotClassificationCurve(given_name, y_test_list_neg, y_test_prob_list_neg, curve_type='pr', data_type='test_data_class0')

            plotCalibrationCurve(given_name, y_all, y_all_prob, data_type='final_train')
            plotCalibrationCurve(given_name, y_test_list, y_test_prob_list, data_type='test')
            plotProbabilityDistribution(given_name, y_all, y_all_prob, data_type='final_train')
            plotProbabilityDistribution(given_name, y_test, y_test_prob, data_type='test')

        elif len(clf.classes_) > 2:
            # loop through classes
            for c in clf.classes_:
                # creating a list of all the classes except the current class
                other_class = [x for x in clf.classes_ if x != c]

                # marking the current class as 1 and all other classes as 0
                y_test_list_ova = [[0 if x in other_class else 1 for x in fold_] for fold_ in y_test_list]
                y_test_prob_list_ova = [[0 if x in c else 1 for x in fold_] for fold_ in y_test_prob_list]

                # Threshold independant
                plotClassificationCurve(given_name, y_all, y_all_prob, curve_type='roc', data_type='final_train')
                plotClassificationCurve(given_name, new_actual_class, new_pred_class, curve_type='roc', data_type=f'test_class_{c}')

                plotClassificationCurve(given_name, y_all, y_all_prob, curve_type='pr', data_type='final_train_class1')
                plotClassificationCurve(given_name, y_all_neg, y_all_prob_neg, curve_type='pr', data_type=f'test_class_{c}')

                plotClassificationCurve(given_name, y_test_list, y_test_prob_list, curve_type='pr', data_type='test_data_class1')
                plotClassificationCurve(given_name, y_test_list_neg, y_test_prob_list_neg, curve_type='pr', data_type='test_data_class0')

                plotCalibrationCurve(given_name, y_all, y_all_prob, data_type='final_train')
                plotCalibrationCurve(given_name, y_test_list, y_test_prob_list, data_type='test')
                plotProbabilityDistribution(given_name, y_all, y_all_prob, data_type='final_train')
                plotProbabilityDistribution(given_name, y_test, y_test_prob, data_type='test')


    elif model_type == 'regression':
        plotYhatVsYSave(given_name, y_test, y_test_pred, data_type='test')
        plotYhatVsYSave(given_name, y_all, y_all_pred, data_type='final_train')

        adjustedR2 = 1 - (1 - clf.score(X_all, y_all)) * (len(y_all) - 1) / (len(y_all) - X_all.shape[1] - 1)
        print('Adjusted R2: {adjustedR2}'.format(adjustedR2=adjustedR2))
        logging.info('Adjusted R2: {adjustedR2}'.format(adjustedR2=adjustedR2))

    # plot the final tree
    # plotTreeStructureSave(clf, given_name)
    featureImportanceSave(clf, given_name + '/feature_importance')

    return clf