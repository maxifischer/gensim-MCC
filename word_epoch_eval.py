from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


to_compare = ["y_pred_emb_a", "y_pred_emb_b_0", "y_pred_emb_b_1", "y_pred_emb_c_0", "y_pred_emb_c_1"]
y_test = np.load("data/y_test.npy")
y_test = [i[1] for i in y_test]
for res in to_compare:
    y_pred = np.load("results/{0}.npy".format(res))
    #print(y_pred[:10])
    y_pred = np.rint(y_pred).astype(int)
    #print(y_pred[:10])
    #print(len(y_pred))
    #print(sum(y_pred))
    #print(y_test[:10])
    print(confusion_matrix(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr)
    plt.savefig("results/roc_{0}.png".format(res))
