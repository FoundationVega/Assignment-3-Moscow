def classify_lda(feature_set, mu_params, cov_params, pi_params, class_labels, threshold_value, compute_crit_value):
    """Classify features_set:

    FOR TASK 1: for the binary classification and one feature you can use explicit expression for
    critical values as decision boundaries; however, you will still need to generalize
    the expressions in slide 49 in Chapter 4 for the case of threshold C different from 1/2 (see slide 51).
    FOR OTHER TASKS: multiple classes and/or multiple features you will have
    to compute the Gaussian discriminant functions delta_r for r = {1,...,k}

    Args:
        feature_set: variable where each row corresponds to an observation or replicate
        to be classified, and each column corresponds to a feature or input variable.

        mu_params: contains the learnt (estimated) mean values for the given
        feature within the classes/groups. In the case of binary classification and 1 feature
        case mu_params should be a vector with two real components.
        Otherwise:
        you can organize them for example as d x k matrix where each j-th column
        (with j = {1,...,k}) contains means for all d features.

        cov_params: contains the learnt (estimated) covariance shared by all the
        the classes/groups (in the 1 feauture case cov_params should contain
        just the variance value).
        Otherwise: covariance matrix in dimension d.

        pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
        for each i-th class (with i = {1,...,k}).
        In the case of binary classification case pi_params should be a vector
        with two nonnegative values with sum = 1.

        class_labels: labels assigned to featuresSet; groupSet will be labeled
        with classLabels.

        threshold_value C: can be specified for the case of binary classification.
        The decision boundary is given by
        \delta_1(x) - \delta_0(x) = log(C/(1-C)) (*) (see slide 51 in Chapter 4)
        and the critical value x_crit is computed as
        as the solution of this equation (*).
        The default value is C=1/2 which renders the decision boundary
        \delta_1(x) - \delta_0(x) = 0
        (hint: be attentive with the log transform,
        log(C/(1-C)) = 0 for the thresshold C = 1/2).

        compute_crit_value: valid only for binary classification and just 1
        feature, is set to 0 by default.

    Returns:
        group_set: contains labels for each of the input feature_set

        x_crit_value: if compute_crit_value is different from 0 contains the
        x_crit_value for a given threshold_value C
    """

    # Determine if it is a binary classification task. If yes, use critical
    # value.
    class_labels_number = len(class_labels)
    num_obs, num_features = feature_set.shape()

    if class_labels_number == 2:
        # in the following code allow for the different thresholds to be set

        # ====================== YOUR CODE STARTS HERE ======================
        pass
        # ====================== YOUR CODE ENDS HERE ========================

        if compute_crit_value and num_features == 1:
            # ====================== YOUR CODE STARTS HERE ======================
            pass
            # ====================== YOUR CODE ENDS HERE ========================
    else:
        # ====================== YOUR CODE STARTS HERE ======================
        pass
        # ====================== YOUR CODE ENDS HERE ========================


def classify_naive_bayes_gauss(feature_set, mu_params, cov_params, pi_params, class_labels, threshold_value):
    """
    Args:
        feature_set: variable where each row corresponds to an observation or replicate
        to be classified, and each column corresponds to a feature or input variable.

        mu_params: contains the learnt (estimated) mean values for the given
        feature within the classes/groups. In the case of binary classification and 1 feature
        case mu_params should be a vector with two real components.
        Otherwise:
        you can organize them for example as d x k matrix where each j-th column
        (with j = {1,...,k}) contains means for all d features.

        cov_params: contains the learnt (estimated) variances of features
        with the classes/groups (in the 1 feature
        case cov_params should contain just the variance value). Otherwise -
        d x k matrix where each j-th column
        (with j = {1,...,k})
        contains variances for all d features

        pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
        for each i-th class (with i = {1,...,k}).
        In the case of binary classification case pi_params should be a vector
        with two nonnegative values with sum = 1.

        class_labels: labels assigned to features_set; group_set will be labeled
        with class_labels.

        threshold_value C: can be specified for the case of binary classification.
        The decision boundary is given by
        \delta_1(x) - \delta_0(x) = log(C/(1-C)) (*) (see slide 51 in Chapter 4)
        and the critical value x_crit is computed as
        as the solution of this equation (*).
        The default value is C=1/2 which renders the decision boundary
        \delta_1(x) - \delta_0(x) = 0
        (hint: be attentive with the log transform,
        log(C/(1-C)) = 0 for the threshold C = 1/2).

    Returns:
        group_set
    """

    class_labels_number = len(class_labels)

    if class_labels_number == 2:
        # in the following code allow for the different thresholds to be set
        # ====================== YOUR CODE STARTS HERE ======================
        pass
        # ====================== YOUR CODE ENDS HERE ========================
    else:
        # ====================== YOUR CODE STARTS HERE ======================
        pass
        # ====================== YOUR CODE ENDS HERE ========================


def classify_qda(feature_set, mu_params, cov_params, pi_params, class_labels, threshold_value):
    """
    Classify featuresSet according to QDA.

    Args:
        feature_set: variable where each row corresponds to an observation or replicate
        to be classified, and each column corresponds to a feature or input variable.

        mu_params: contains the learnt (estimated) mean values for the given
        feature within the classes/groups. In the case of binary classification and 1 feature
        case mu_params should be a vector with two real components.
        Otherwise:
        you can organize them for example as d x k matrix where each j-th column
        (with j = {1,...,k}) contains means for all d features.

        cov_params: contains the learnt (estimated) covariances for each of
        the k classes/groups (in the 1 feature
        case cov_params should contain just the k variance values). Otherwise -
        covariance matrices in dimension d. You can think of creating a cell
        array or a three-dimensional array

        pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
        for each i-th class (with i = {1,...,k}).
        In the case of binary classification case pi_params should be a vector
        with two nonnegative values with sum = 1.

        class_labels: labels assigned to featuresSet; group_set will be labeled
        with class_labels.

        threshold_value C: can be specified for the case of binary classification.
        The decision boundary is given by
        \delta_1(x) - \delta_0(x) = log(C/(1-C)) (*) (see slide 51 in Chapter 4)
        and the critical value x_crit is computed as
        as the solution of this equation (*).
        The default value is C=1/2 which renders the decision boundary
        \delta_1(x) - \delta_0(x) = 0
        (hint: be attentive with the log transform,
        log(C/(1-C)) = 0 for the thresshold C = 1/2).

    Returns:
        group_set
    """


    class_labels_number = len(class_labels)
    num_obs, num_features = feature_set.shape

    delta_vals = np.zeros((num_obs, class_labels_number))

    # compute the delta_vals
    # ====================== YOUR CODE STARTS HERE ======================
    pass
    # ====================== YOUR CODE ENDS HERE ========================

    # if one has only two classes you can chose a different threshold value
    # (other than 1)

    if class_labels_number == 2 and threshold_value != 0.5:
        # ====================== YOUR CODE STARTS HERE ======================
        pass
        # ====================== YOUR CODE ENDS HERE ========================
    else:
        # ====================== YOUR CODE STARTS HERE ======================
        pass
        # ====================== YOUR CODE ENDS HERE ========================


def compute_loss(Y, Y_pred, type_of_loss):
    """
    Compute loss

    Returns:
        loss
        type_I_error
        type_II_error
    """

    loss = 0

    # relevant only for binary classification. In this case the labels are
    # assumed to be 1 and 0.
    type_I_error = 0
    type_II_error = 0

    if type_of_loss == 'mse':
        # ====================== YOUR CODE STARTS HERE ====================
        pass
        # ====================== YOUR CODE ENDS HERE ======================
    elif type_of_loss == '0-1':
        if set(Y_pred) == {0, 1}:
            # ====================== YOUR CODE STARTS HERE ====================
            pass
            # ====================== YOUR CODE ENDS HERE ======================
        else:
            # ====================== YOUR CODE STARTS HERE ====================
            pass
            # ====================== YOUR CODE ENDS HERE ======================
    else:
        raise ValueError('The type of loss is not known')


def fit_lda(feature_set, group_set, class_labels):
    """
    Args:
        feature_set: where each row corresponds to an observation or replicate,
        and each column corresponds to a feature or variable

        group_set: variable with each row representing a class label.
        Each element of groupSet specifies the group of the corresponding row of
        feature_set

        class_labels: provide the labels according to which we order the output
        parameters. They determine in which order the parameters are constructed

    Returns:
        mu_params: contains the learnt (estimated) mean values for the given
        feature within the classes/groups. In the case of binary classification and 1 feature
        case mu_params should be a vector with two real components. Otherwise -
        you can organize them for example as d x k matrix where each j-th column
        (with j = {1,...,k}) contains means for all d features

        cov_params: contains the learnt (estimated) covariance shared by all the
        the classes/groups (in the 1 feature case cov_params should contain just
        the variance value). Otherwise - covariance matrix in dimension d

        pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
        for each i-th class (with i = {1,...,k}).
        In the case of binary classification
        case pi_params should be a vector with two nonnegative values with sum = 1
    """

    #  Instructions
    #  ------------
    #  Note that these are supervised learning algorithms, thus:
    #   - pi_params are sample-based estimates of prior probabilities of each class;
    #   - mu_params are means of features within teh classes;
    #   - cov_params is the pooled mean of the var/covariance matrix shared by
    #     all classes (see slide 43 in Chapter 4);

    # ====================== YOUR CODE STARTS HERE ======================
    pass
    # ====================== YOUR CODE ENDS HERE ========================


def fit_naive_bayes_gauss(feature_set, group_set, class_labels):
    """
    Args:
        feature_set: where each row corresponds to an observation or replicate,
        and each column corresponds to a feature or variable

        group_set: variable with each row representing a class label.
        Each element of groupSet specifies the group of the corresponding row of
        feature_set

        class_labels: provide the labels according to which we order the output
        parameters. They determine in which order the parameters are constructed

    Returns:
        mu_params: contains the learnt (estimated) mean values for the given
        feature within the classes/groups. In the case of binary classification and 1 feature
        case mu_params should be a vector with two real components. Otherwise -
        you can organize them for example as d x k matrix where each j-th column
        (with j = {1,...,k})
        contains means for all d features

        cov_params: contains the learnt (estimated) variances of feautures
        with the classes/groups (in the 1 feature
        case cov_params should contain just the variance value). Otherwise -
        d x k matrix where each j-th column
        (with j = {1,...,k})
        contains variances for all d features

        pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
        for each i-th class (with i = {1,...,k}).
        In the case of binary classification
        case pi_params should be a vector with two nonnegative values with sum = 1
    """

    #  Instructions
    #  ------------
    #  Note that these are supervised learning algorithms, thus:
    #   - pi_params are sample-based estimates of prior probabilities of each class;
    #   - mu_params are means of features within classes;
    #   - cov_params are only the variances of the var/covariance matrices of
    #     each feauture within each class. Think carefully on how to fill the
    #     'cov_params' and also remember the conditional independence
    #     assumptions adopted for Naive Bayes
    #

    # ====================== YOUR CODE STARTS HERE ======================
    pass
    # ====================== YOUR CODE ENDS HERE ========================


def fit_qda(feature_set, group_set, class_labels):
    """
    Args:
        feature_set: where each row corresponds to an observation or replicate,
        and each column corresponds to a feature or variable

        group_set: variable with each row representing a class label.
        Each element of groupSet specifies the group of the corresponding row of
        feature_set

        class_labels: provide the labels according to which we order the output
        parameters. They determine in which order the parameters are constructed

    Returns:
        mu_params: contains the learnt (estimated) mean values for the given
        feature within the classes/groups. In the case of binary classification and 1 feature
        case mu_params should be a vector with two real components. Otherwise -
        you can organize them for example as d x k matrix where each j-th column
        (with j = {1,...,k})
        contains means for all d features

        cov_params: contains the learnt (estimated) covariances for each of
        the k classes/groups (in the 1 feature
        case cov_params should contain just the k variance values). Otherwise -
        covariance matrices in dimension m. You can think of creating a cell
        array or a three-dimensional array

        pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
        for each i-th class (with i = {1,...,k}).
        In the case of binary classification
        case pi_params should be a vector with two nonnegative values with sum = 1
    """

    #  Instructions
    #  ------------
    # See slide 43 in Chapter 4
    #  Note that these are supervised learning algorithms, thus:
    #   - pi_params are sample-based estimates of prior probabilities of each class;
    #   - mu_params are means of features within the classes;
    #   - cov_params are the var/covariance matrices of each class. Think
    #     carefully on how to fill the 'cov_params'!
    #

    # ====================== YOUR CODE STARTS HERE ======================
    pass
    # ====================== YOUR CODE ENDS HERE ========================


#  Generative classifiers for binary and multiclass classification with one
#  and multiple features
#
#  Instructions
#  ------------
#
#  This file contains code that helps you to get started on the
#  generative classification exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     fit_lda
#     classify_lda
#     fit_qda
#     classify_qda
#     compute_loss
#     fit_naive_bayes_gauss
#     classify_naive_bayes_gauss
#
#  Data description (see http://archive.ics.uci.edu/ml/datasets/Wine)
#  ------------------------------------------------------------------
#
#  > These data are the results of a chemical analysis of
#    wines grown in the same region in Italy but derived from three
#    different cultivars.
#    The analysis determined the quantities of 13 constituents
#    found in each of the three types of wines.
#
#    Number of Instances
#
#         class 1 -> 59
# 	      class 2 -> 71
# 	      class 3 -> 48
#
#    Number of Attributes
#
#         13
#
## Load Data ======================================================


#  Load the 'wine.csv' dataset and determine how many classes there are in
#  the dataset. Create separate variables containing the class labels and all the
#  available features. Create a variable containing the names of the features,
#  for that look at the description of the data following the link provided
#  above. Determine how many representatives of each class
#  there are in the dataset



## Part I: Binary Classification with One Feature ========================

# Select only classes 1 and 3 for this part and feature 'Proanthocyanins'.
# In this binary classification exercise assign label 0 to Class 1 and
# label 1 to Class 3.


###########################################################################
# Task 0: Plot the data by creating two <count> density-normalized histograms in
# two different colors
# of your choice; for that use the specific normalization and bin width 0.3.
# Add the grid.
# Add descriptive legend and title.
###########################################################################


###########################################################################
# Task 1 : Construct LDA classifier. For that fill in the function fit_lda
# and classify_lda. Both functions should be constructed in order to
# work with multiple classes and multiple feautures if needed. We start
# here however with only binary classification which for the case of univariate feautures
# admits the explicit critical decision boundary value x_crit (see Slide 49 in Chapter 4),
# since we assume that the threshold is C = 1/2 (see our discussions in Tutorial 3)
##########################################################################


###########################################################################
# Task 2 : Compute the empirical value of the error using the 0-1 loss.
# For that add typeOfLoss '0-1' option to the function compute_loss from the
# previous assignment. Additionally, this function needs to output the Type I and Type II
# errors (false positive and false negative) which will be used only in the case of
# binary classification
##########################################################################


###########################################################################
# Task 3 : Plot the resulting classification.
# Create two histograms in two different colors of your choice: for these,
# use normalization, and bin width 0.3.
# Superimpose the two normal distributions and the Gaussian mixture distribution that you
# obtain with the parameters computed in the 'fit_lda' function.
# Add the grid.
# Add descriptive legend and title.
# Plot the decision boundary (critical value x_crit for the given threshold C of interest,
# which is set by default to 1/2)
##########################################################################


###########################################################################
# Task 4 : Construct QDA classifier. For that fill in the function fit_qda
# and classify_qda. Both functions should be constructed in order to
# work with multiple classes and multiple features if needed. We start
# here however with only two-classes classification first.
##########################################################################


###########################################################################
# Task 5 : Compute the empirical value of the error using the 0-1 loss.
# using the function compute_loss together with the Type I and Type II
# errors
##########################################################################


###########################################################################
# Task 6 : Plot the resulting classification.
# Create two histograms in two different colors of your choice: for these,
# use normalization and bin width 0.3.
# Superimpose the two normal distributions and the mixed distribution that you
# obtain as a result from the 'fit_lda' function.
# Add the grid.
# Add descriptive legend and title.
##########################################################################


###########################################################################
# Task 7 : Construct the Gaussian Naive Bayes classifier. For that fill in the
# function fit_naive_bayes_gauss and classify_naive_bayes_gauss.
# Both functions should be constructed in order to work with multiple
# classes and multiple features if needed. However, we start with only
# two-classes classification.
#########################################################################


###########################################################################
# Task 8 : Compute the empirical value of the error using the 0-1 loss.
# using the function compute_loss together with the Type I and Type II
# errors
##########################################################################


###########################################################################
# Task 9 : Plot the resulting classification.
# Create two histograms in two different colors  of your choice: for these,
# use normalization and bin width 0.3.
# Superimpose the two normal distributions and the mixed distribution that you
# obtain as a result from the 'fit_naive_bayes_gauss' function.
# Add the grid.
# Add descriptive legend and title.
##########################################################################


## Part II: Binary Classification with Two Features ======================

# Select only classes 1 and 3 for this part and features:
#
#   - 'Proanthocyanins'
#   - 'Alcalinity of ash'
#
# In this binary classification exercise assign label 0 to Class 1 and
# label 1 to Class 3.


###########################################################################
# Task 10 : Generalize LDA, QDA, GNB to the case of two features. Report
# their training/LOOCV errors for the stadard threshold C=1/2, plot.
##########################################################################


###########################################################################
# Plot the resulting classification.
# Add the grid.
# Add descriptive legend and title.
# Mark misclassified observations.
##########################################################################

# Compare all the classifiers via plotting their ROC curves and computing the area under ROC (AUC).
# Draw conclusions.


## Part III: 3-Classes Classification with Many Features ====================

# Select only classes 1, 2, and 3 for this part and features:
#
#   - 'Alcohol'
#   - 'Flavanoids'
#   - 'Proanthocyanins'
#   - 'Color intensity'
#


###########################################################################
# Task 11 : Construct QDA classifier for the following:
#
#   - 'Alcohol'
#   - 'Alcohol' + 'Proanthocyanins'
#   - All features listed above
#
##########################################################################


###########################################################################
# Compute the empirical value of the errors using the 0-1 loss.
##########################################################################

