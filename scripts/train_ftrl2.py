from model import *

# A, paths
submission = 'submission1234.csv'  # path of to be outputted submission file

# B, model
alpha = .05  # learning rate
beta = .1   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
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

query = "SELECT trainSearchStream.* \
FROM trainSearchStream \
WHERE ObjectType=3 LIMIT 500000;"

for t, date, ID, x, y in data(query, engine, D):  # data is a generator
    #    t: just a instance counter
    # date: you know what this is
    #   ID: id provided in original data
    #    x: features
    #    y: label (click)

    # step 1, get prediction from learner
    p = learner.predict(x)
    loss += logloss(p, y)
    learner.update(x, p, y)
    if t %10000 == 0:
        print "{0:f}".format(loss/10000.)
        loss = 0.

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
        p = "{0:f}"
        outfile.write('%s,%s\n' % (ID, p))