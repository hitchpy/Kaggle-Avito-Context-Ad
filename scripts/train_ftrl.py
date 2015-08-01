from model import *

# A, paths
submission = 'submission_yu.csv'  # path of to be outputted submission file

# B, model
alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
holdafter = False #datetime.strptime("2015-04-26", "%Y-%m-%d")   # data after date N (exclusive) are used as validation
#holdout = None  # use every N training instance for holdout validation

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
engine = sa.create_engine('sqlite:///../input/database.sqlite')
loss = 0.
count = 0
#"2015-04-25","2015-04-26", "2015-04-27","2015-04-28","2015-04-29","2015-04-30","2015-05-01","2015-05-02","2015-05-03","2015-05-04","2015-05-05","2015-05-06","2015-05-07","2015-05-08","2015-05-09","2015-05-10",
for d in ["2015-05-11","2015-05-12","2015-05-13","2015-05-14","2015-05-15","2015-05-16","2015-05-17","2015-05-18", "2015-05-19","2015-05-20"]:
    query = "SELECT trainSearchStream.*, AdsInfo.CategoryID AS acid, \
    Price, date(SearchDate) AS SearchDate, IsUserLoggedOn, SearchQuery, SearchInfo.LocationID, \
    SearchInfo.CategoryID, UserAgentOSID, UserDeviceID, c1.Level AS slevel, c2.Level AS alevel \
    FROM trainSearchStream JOIN SearchInfo ON trainSearchStream.SearchID=SearchInfo.SearchID \
    JOIN AdsInfo ON AdsInfo.AdID=trainSearchStream.AdID JOIN Category c1 ON c1.CategoryID=SearchInfo.CategoryID \
    JOIN UserInfo ON UserInfo.UserID=SearchInfo.UserID JOIN Category c2 ON c2.CategoryID=AdsInfo.CategoryID \
    WHERE ObjectType=3 AND date(SearchInfo.SearchDate)='" + d + "';"

    for t, date, ID, x, y in data(query, engine, D):  # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)

        # step 1, get prediction from learner
        p = learner.predict(x)

        if (holdafter and datetime.strptime(date, "%Y-%m-%d") > holdafter):
            # step 2-1, calculate validation loss
            #           we do not train with the validation data so that our
            #           validation loss is an accurate estimation
            #
            # holdafter: train instances from day 1 to day N
            #            validate with instances from day N + 1 and after
            #
            # holdout: validate with every N instance, train with others
            loss += logloss(p, y)
            count += 1
        else:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y)
        if t %10000 == 0:
            print t

#print('Date %s finished, validation logloss: %f, elapsed time: %s' % (
#d, loss/count, str(datetime.now() - start)))

print "start testing"
##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
query = "SELECT testSearchStream.*, AdsInfo.CategoryID AS acid, \
    Price, date(SearchDate) AS SearchDate, IsUserLoggedOn, SearchQuery, SearchInfo.LocationID, \
    SearchInfo.CategoryID, UserAgentOSID, UserDeviceID, c1.Level AS slevel, c2.Level AS alevel \
    FROM testSearchStream LEFT JOIN SearchInfo ON testSearchStream.SearchID=SearchInfo.SearchID \
    LEFT JOIN AdsInfo ON AdsInfo.AdID=testSearchStream.AdID LEFT JOIN Category c1 ON c1.CategoryID=SearchInfo.CategoryID \
    LEFT JOIN UserInfo ON UserInfo.UserID=SearchInfo.UserID LEFT JOIN Category c2 ON c2.CategoryID=AdsInfo.CategoryID \
    WHERE ObjectType=3;"
with open(submission, 'w') as outfile:
    outfile.write('ID,IsClick\n')
    for t, date, ID, x, y in data(query, engine, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p)))
