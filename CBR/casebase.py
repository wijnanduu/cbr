import operator
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from termcolor import colored
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from copy import deepcopy

###########################
### Auxiliary functions ###
###########################

# Recursive definition of the transitive closure. 
def Rplus(x, R):
    return R[x] | {z for y in R[x] for z in Rplus(y, R)}

# Takes as input a (finite) Hasse Diagram, where the nodes are given by A and 
# the covering relation by R, and return the reflexive transitive closure of R. 
def tr_closure(A, R):
    R  = {x : {y for y in A if (x, y) in R} for x in A}
    return {(x, x) for x in A} | {(x, y) for x in A for y in Rplus(x, R)}

# Define <=_s and <_s in terms of the <= and < relations.
# These are needed for the case class.  
def le(s, v1, v2):
    return v1 <= v2 if s == 1 else v1 >= v2
def lt(s, v1, v2):
    return v1 < v2 if s == 1 else v1 > v2

###############
### Classes ###
###############

# A dimension is a partially ordered set. 
# This order must be specified at creation by the 'le' function.
# Its elements are assumed to be partially ordered by <=, specified
# by the 'le' function, where the default direction is for the plaintiff. 
class dim():
    def __init__(self, name, le):
        self.name = name
        self.le = le if type(le) != set else lambda x, y: (x, y) in le
    def __eq__(self, d):
        return self.name == d.name and self.le == d.le 
    def __repr__(self):
        return f"dim({self.name})"

# A coordonate is a value in some dimension. 
# They can be compared using the partial order of the dimension they belong to.
class coord():
    def __init__(self, value, dim):
        self.dim = dim
        self.value = value
    def __le__(self, c2):
        return self.dim.le(self.value, c2.value)
    def __lt__(self, c2):
        return self != c2 and self.dim.le(self.value, c2.value)
    def __ge__(self, c2):
        return self.dim.le(c2.value, self.value)
    def __gt__(self, c2):
        return self != c2 and self.dim.le(c2.value, self.value)
    def __str__(self):
        return str(f"{self.value}")
    def __repr__(self):
        return str(f"{self.value}")
    def __eq__(self, value):
        return self.value == value

# A class for cases, i.e. fact situations together with an outcome.
# A fact situation is represented as a dictionary mapping the dimensions to 
# a coordonate in that dimension.  
class case():
    def __init__(self, F, s):
        self.F = F 
        self.s = s
    def __le__(self, d):
        return not any(self.diff(d.F))
    def __getitem__(self, key):
        return self.F[key]
    def __setitem__(self, key, value):
        self.F[key] = value
    def __eq__(self, c):
        return self.F == c.F and self.s == c.s
    def __str__(self):
        table = tabulate(
            [[d, self[d].value] for d in self.F], 
            showindex=False, 
            headers=["d", "c[d].value"], 
            colalign=("left", "left")
            )
        sep = len(table.split('\n')[1]) * "-"
        return sep + '\n' + table + '\n' + sep + '\n' + f"Outcome: {self.s}" + '\n' + sep
    def diff(self, G):
        for d in self.F:
            if not le(self.s, self[d], G[d]):
                yield d

# A class for a case base, in essence it is just a list of cases
# but it has a custom init and extra functions. 
class casebase(list):
    """
    Inputs.
        csv: 
            The file path to the (processed) csv file. 
        catcs: 
            A list of the categorical columns in the data. This will
            also determine a variable 'ordcs' of ordinal columns. If 
            no value is provided it will automatically be defined 
            as those columns of which not all values can be converted to
            integers. 
        replace: 
            A boolean indicating whether the values in the dataframe
            should be replaced based on their order. For example,
            if a column has possible values 'a', 'b', and 'c', with
            increasing correlation with the 'Label' variable respectively,
            then we can simply replace them with 0, 1, and 2 respectively
            together with the usual less-than-or-equal order. If this
            variable is false then the values are not replaced and instead
            the dimension is defined using a set indicating their order. 
        manords: 
            Allows the user to provide a dictionary mapping columns
            names to desired orders, thereby overriding the default
            behaviour which assigns order based on the 'method' value. 
        verb: 
            A boolean specifying whether the function should be verbose.
        method: 
            A string indicating the desired method of determining the dimension orders.
        size:
            Limits the size to the specified integer, if possible. 
        shuffle:
            Shuffles the input csv before creating the case base. 
            This is useful in combination with size when the labels are distributed unevenly.
            Since in most other cases it makes no differences it is on by default. 

    Attributes.
        df: The dataframe holding the csv. 
        D: A dictionary mapping names of dimensions to a dimension class object.
        F: The forcing relation on cases. Needs to be initialized at some point. 
        Fd: A dictionary version of the forcing relation.
        Fid: A dictionary version of the inverse of the forcing relation.  
    """
    def __init__(self, csv, catcs=None, replace=False, manords={}, verb=False, method='logreg', size=-1, shuffle=True):
        # Read the csv file and create a list holding the column names. 
        df = pd.read_csv(csv)
        assert "Label" in df.columns.values, "There is no 'Label' column in the csv data."
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        cs = [c for c in df.columns.values if c != "Label"]
        self.df = df
        self.cs = cs

        # Identify the categorical and ordinal columns (if this hasn't been done yet)
        # by trying to convert the values of the column to integers.
        if catcs == None:
            catcs = []
            ordcs = []
            for c in cs:
                try:
                    df[c].apply(int)
                    ordcs += [c]
                except ValueError:
                    catcs += [c]
        else:
            ordcs = [c for c in cs if c not in catcs]

        # Initialize the dimensions using the manually specified ones. 
        self.D = {d : dim(d, manords[d]) for d in manords}

        # Remove the columns that are manually specified. 
        catcs = [c for c in catcs if c not in manords]
        ordcs = [c for c in ordcs if c not in manords]

        # Determine the order based on a method that works with a 'coefficient function'.
        if method == 'pearson' or 'logreg':

            # Compute the coefficient dictionary based on either the pearson or logreg method.
            if method == 'pearson':
                coeffs = pd.get_dummies(df.drop(manords, axis='columns')).corr()["Label"]
            elif method == 'logreg':
                X = pd.get_dummies(df.drop(manords, axis='columns').drop("Label", axis='columns')).to_numpy()
                dcs = pd.get_dummies(df[cs].drop(manords, axis='columns')).columns.values
                y = df["Label"].to_numpy()
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)
                clf = LogisticRegression(random_state=0).fit(X, y)
                coeffs = dict(zip(dcs, clf.coef_[0]))

            # Determine orders of ordinal features using the coeffs dict. 
            self.D.update({c : dim(c, operator.le) if coeffs[c] > 0 else dim(c, operator.ge) for c in ordcs})

            # Log the orders of the ordinal features. 
            if verb:
                print("\nPrinting dimension orders.")
                for c in ordcs:
                    print(f"{colored(c, 'red')}: {colored('Ascending' if coeffs[c] > 0 else 'Descending', 'green')} ({round(coeffs[c], 2)})")

            # Determine orders of categorical features using the coeffs dict.
            self.tl = {}
            for c in catcs:
                cvals = df[c].unique()
                scvals = sorted(cvals, key=lambda x: coeffs[f'{c}_{x}'])

                # Log the orders of this feature.
                if verb:
                    print(f"{colored(c, 'red')}: " + " < ".join([f"{colored(v, 'green')} ({round(coeffs[f'{c}_{v}'], 4)})" for v in scvals]))

                # Replace the values of the categorical feature with numbers, so that 
                # we can simply compare using <= on the naturals, if enabled. 
                if replace:
                    for i, val in enumerate(scvals):
                        df[c] = df[c].replace(val, i)
                    self.D[c] = dim(c, operator.le)

                    # Create a dict for translating the old value names to the new ones. 
                    self.tl[c] = dict(zip(cvals, [scvals.index(val) for val in cvals]))
                    if verb:
                        print(f"Values for column {c} were replaces as follows:")
                        print(tabulate([[val, self.tl[c][val]] for val in cvals], headers=["orig", "new"], tablefmt="simple_outline"))

                # Otherwise, make the relation on the original categorical values.
                else:
                    hd = {(scvals[i], scvals[i+1]) for i in range(len(scvals) - 1)}
                    trhd = tr_closure(df[c].unique(), hd)
                    self.D[c] = dim(c, trhd)

        # Read the rows into a list of cases. 
        cases = [case({d : coord(r[d], self.D[d]) for d in self.D}, r["Label"]) for _, r in df.iterrows()]

        # Reduce the size to the desired number, if set. 
        if size != -1:
            cases = cases[:size]

        # Call the list init function to load the cases into the CB. 
        super(casebase, self).__init__(cases)

        # Initialize some variables used by the statistics functions. 
        self.inds = range(len(df.index))
        self.adf = pd.DataFrame() 
        self.forcing_initialized = False 
        self.consistency_initialized = False
        self.landmarks_initialized = False
        self.bcitability_PR_initialized = False
        self.bcitability_PRp_initialized = False
        self.bcitability_WGPV_initialized = False

        # Separate the cases into two lists based on outcome. 
        self.CB0 = [c for c in self if c.s == 0]
        self.CB1 = [c for c in self if c.s == 1]
        self.iCB0 = [i for i in self.inds if self[i].s == 0]
        self.iCB1 = [i for i in self.inds if self[i].s == 1]
        self.iCBs = {s : self.iCB0 if s == 0 else self.iCB1 for s in [0, 1]}

    # @property
    def init_forcing(self):
        """
        Initializes the forcing relation on cases and related information, used by the other analysis functions. 
        """
        inds = self.inds        

        # Compute all forcing relations between the cases.
        print("Computing the forcing relation on cases.")
        self.F = {(i, j) for i in tqdm(inds) for j in inds if self[i] <= self[j]}
        self.Fd = {i : [] for i in inds}
        self.Fid = {i : [] for i in inds}
        for i, j in self.F:
            self.Fd[i] += [j]
            self.Fid[j] += [i]

        # Separate from F the forcings that lead to inconsistency.
        self.I = {(i, j) for (i, j) in self.F if self[i].s != self[j].s}
        self.Id = {i : set() for i in inds}
        for i, j in self.I:
            self.Id[i] |= {j}
            self.Id[j] |= {i}

        # Calculate the forcing scores for all cases, 
        # which is the number of cases forced. 
        self.adf["Scores"] = [len(self.Fd[i]) for i in inds]
        self.adf["Score (same outcome)"] = [len(self.Fd[i]) - len(self.Id[i]) for i in inds]
        self.adf["Score (diff outcome)"] = [len(self.Id[i]) for i in inds]
        self.adf["Label"] = [self[i].s for i in inds]

        # Set the flag.
        self.forcing_initialized = True

    # Calculate the consistency statistic.
    def init_consistency(self):
        if not self.forcing_initialized:
            self.init_forcing()

        self.adf["Consistency"] = [int(len(self.Id[i]) == 0) for i in self.inds]

        # Set the flag. 
        self.consistency_initialized = True

    # Print CB's consistency percentage.
    def report_consistency(self):
        if not self.consistency_initialized:
            self.init_consistency()

        inc0 = len(self.CB0) - sum(self.adf[self.adf["Label"] == 0]["Consistency"])
        inc1 = len(self.CB1) - sum(self.adf[self.adf["Label"] == 1]["Consistency"])
        print(f"The consistency is {sum(self.adf['Consistency'])} / {len(self)} = {((sum(self.adf['Consistency']))/(len(self)))*100}%.")
        print(f"Of the {len(self.CB0)} cases with label 0 there are {inc0} inconsistent ones.")
        print(f"Of the {len(self.CB1)} cases with label 1 there are {inc1} inconsistent ones.")

    # Calculates landmark related info. 
    def init_landmarks(self):
        if not self.forcing_initialized:
            self.init_forcing()

        # Make a dictionary holding all landmarks.
        self.Ls = {s : [i for i in self.inds if self[i].s == s and not any(self[i].s == self[j].s for j in self.Fid[i] if i != j)] for s in [0, 1]}
        self.adf["Landmark"] = [i in self.Ls[self[i].s] for i in self.inds]

        # Calculate the two top forcing landmarks l0 and l1. 
        adfl0 = self.adf[(self.adf["Label"] == 0) & (self.adf["Landmark"] == True)].sort_values("Scores", ascending=False)
        adfl1 = self.adf[(self.adf["Label"] == 1) & (self.adf["Landmark"] == True)].sort_values("Scores", ascending=False)
        il0 = adfl0.index[0]
        il1 = adfl1.index[0]
        self.l0 = self[il0]
        self.l1 = self[il1]

        # Set the flag.
        self.landmarks_initialized = True
        
    # Print the landmark information.
    def report_landmarks(self):
        if not self.landmarks_initialized:
            self.init_landmarks()
        
        # Print the top two forcing landmarks. 
        print("Top forcing landmark with outcome 0:")
        print(self.l0)
        print("Top forcing landmark with outcome 1:")
        print(self.l1)

        # Show how many landmarks there are for either class.
        print(f"\nNumber of landmarks for outcome 0: {len(self.Ls[0])}")
        print(f"Number of landmarks for outcome 1: {len(self.Ls[1])}")

        # Compute the number of ordinary cases.
        print("\nNumber of ordinary (aka trivial) cases for either class:")
        print(f"For outcome 0: {len(self.CB0)} - {len(self.Ls[0])} = {len(self.CB0) - len(self.Ls[0])}")
        print(f"For outcome 1: {len(self.CB1)} - {len(self.Ls[1])} = {len(self.CB1) - len(self.Ls[1])}")

    # Calculate the number of best citable precedents according to PR21.
    # This is calculated according to the formal definition in PR21. 
    # The algorithm used here for calculating this statistic is suboptimal.
    # A more refined way to compute it is described in the paper:
    # 'A simple sub-quadratic algorithm for computing the subset partial order'. 
    def init_bcitability_PR(self):
        if not self.forcing_initialized:
            self.init_forcing()

        Citability = []
        D = self.D

        # Define the relevant differences. 
        def rel_diffs(i, j):
            p = self[i]
            f = self[j]
            return {(d, str(p[d])) for d in D if not le(p.s, p[d], f[d])}

        print("Computing best citability information.")
        for i in tqdm(self.inds):
            citable = [j for j in self.iCBs[self[i].s] if i != j and any(le(self[i].s, self[j][d], self[i][d])for d in self.D)]
            diffs = {j : rel_diffs(j, i) for j in citable}
            best_citable = [j for j in citable if not any(diffs[k] < diffs[j] for k in citable)]
            Citability += [len(best_citable)]
        self.adf["Citability_PR"] = Citability

        # Set the flag.
        self.bcitability_PR_initialized = True

    # Print the citability information according to PR21 (strictly speaking). 
    def report_bcitability_PR(self):
        if not self.bcitability_PR_initialized:
            self.init_bcitability_PR()

        desc = self.adf['Citability_PR'].describe()
        print(f"[PR] Mean and std. of best citability is {round(desc['mean'], 1)} ± {round(desc['std'], 1)}.")

    # Calculate the number of best citable precedents according to PR21.
    # This is calculated according to the implicit definition in PR21.
    # (I.e. the one used to calculate the percentages shown in Table 5.)
    def init_bcitability_PRp(self):
        Citability = []
        D = self.D

        # Define the relevant differences. 
        def rel_diffs(i, j):
            p = self[i]
            f = self[j]
            return {d for d in D if not le(p.s, p[d], f[d])}

        print("Computing best citability information.")
        for i in tqdm(self.inds):
            citable = [j for j in self.iCBs[self[i].s] if i != j and any(le(self[i].s, self[j][d], self[i][d])for d in self.D)]
            diffs = {j : rel_diffs(j, i) for j in citable}
            best_citable = [j for j in citable if not any(diffs[k] < diffs[j] for k in citable)]
            Citability += [len(best_citable)]
        self.adf["Citability_PRp"] = Citability

        # Set the flag.
        self.bcitability_PRp_initialized = True

    # Print the citability information according to PR21 (not strictly speaking). 
    def report_bcitability_PRp(self):
        if not self.bcitability_PRp_initialized:
            self.init_bcitability_PRp()

        desc = self.adf['Citability_PRp'].describe()
        print(f"[PRp] Mean and std. of best citability is {round(desc['mean'], 1)} ± {round(desc['std'], 1)}.")

    # Calculate the number of best citable precedents according to WGPV22.
    def init_bcitability_WGPV(self):
        Citability = []
        # D = {d for d in self[0].F}
        D = self.D
        print("Computing best citability information.")
        for i in tqdm(self.inds):
            s = self[i].s
            ca = {j for j in self.iCBs[s] if i != j and any(le(s, self[i].F[d], self[j].F[d]) for d in D)}
            scores = defaultdict(lambda: [])
            for j in ca:
                v = len(set(self[j].diff(self[i])))
                scores[v] += [j]
            cb = {j for j in scores[min(scores)]}
            scores = defaultdict(lambda: [])
            for j in cb:
                v = len(set(d for d in D if self[i][d] == self[j][d]))
                scores[v] += [j]
            cc = {j for j in scores[max(scores)]}
            Citability += [len(cc)]
        self.adf["Citability_WGPV"] = Citability

        # Set the flag.
        self.bcitability_WGPV_initialized = True

    # Print the citability information according to WGPV22.
    def report_bcitability_WGPV(self):
        if not self.bcitability_WGPV_initialized:
            self.init_bcitability_WGPV()
            
        desc = self.adf['Citability_WGPV'].describe()
        print(f"[WGPV] Mean and std. of best citability is {round(desc['mean'], 1)} ± {round(desc['std'], 1)}.")

    # Calculate and print the minimum (?) number of deletions before CB is consistent.
    def report_minimumdeletions(self):
        if not self.forcing_initialized:
            self.init_forcing()

        removals = 0
        Id = deepcopy(self.Id)
        while sum(s := [len(Id[i]) for i in self.inds]) != 0:
            k = np.argmax(s)
            for i in self.inds:
                Id[i] -= {k}
            Id[k] = set()
            removals += 1

        # Print the resulting statistic.
        print(f"Removals to obtain cons.: {removals}/{len(self)} = {round(removals/len(self)*100, 1)}%")

    # Bundles some report functions to analyze the CB. 
    def analyze(self):
        self.report_consistency()
        # self.report_bcitability_PR()      
        # self.report_bcitability_PRp()      
        # self.report_bcitability_WGPV()       
        self.report_landmarks()
        self.report_minimumdeletions()

    # A function which pretty prints a comparison between cases a and b.
    # Mostly useful for debugging purposes. 
    def compare(self, a, b):
        # Compare two values v,w according to their dimensions order and return the result. 
        def R(v, w):
            if v == w:
                return "="
            if v < w:
                return "<"
            elif w < v:
                return ">"
            else:
                return "|"

        # Form a pandas dataframe holding the comparison data.
        compdf = pd.DataFrame(
            {
                "a" : [a[d] for d in self.D] + [a.s], 
                "R" : [R(a[d], b[d]) for d in self.D] + [""],
                "b" : [b[d] for d in self.D] + [b.s],
                "F" : ["X" if d in a.diff(b) else "" for d in self.D] + [""]
            }, 
            index = list(self.D.keys()) + ['Label'])

        # Pretty print this dataframe using the tabulate package. 
        print(tabulate(
            compdf, 
            showindex=True, 
            headers=["Dimension"] + list(compdf.columns), 
            colalign=("left", "right", "center", "left", "center")
            ))

