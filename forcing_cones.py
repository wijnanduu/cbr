from CBR.casebase import *
from matplotlib import pyplot as plt

# Set latex parameters. 
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'times',
        #   'text.latex.unicode': True,
          'text.latex.preamble' : r'\usepackage{lmodern}\usepackage{mathptmx}'
          }
plt.rcParams.update(params) 

# The plot made this code will be saved as '{save}.pdf' if save is not an empty string,
# if save is an empty string then the output will not be saved
save = ""
# save = "cones"

# Load the case base and its dataframe. 
CB = casebase(
    "data/corels.csv",
    verb=True,
    method='logreg',
    )
df = CB.df
CB.init_forcing()
CB.init_landmarks()

# Colors for the outcomes. 
c0 = 'green'
c1 = 'red' 

# Create the plot
fig, axes = plt.subplots()

# Fill the plot with the data
plt.scatter(
    df['Age'], 
    df['Priors'], 
    c = [c0 if b == 0 else c1 for b in df['Label']], 
    # alpha = .15, 
    alpha = .15, 
    s = 15,
    marker = 'o',
    )

# Set plot ranges, and labels. 
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'\texttt{Priors}')
plt.xlabel(r'\texttt{Age}')

# Add special indicators for the landmarks with outcome 0.
ages0 = [CB[l]["Age"].value for l in CB.Ls[0]]
priors0 = [CB[l]["Priors"].value for l in CB.Ls[0]]
plt.scatter(
    ages0, # Age
    priors0, # Priors
    c = c0,
    s = 15,
    marker = 'x',
)

# Visualize the forcing cones of the landmarks with outcome 0.
alph = 0.1
xd = np.array([xmin, xmax])
for a, p in zip(ages0, priors0):
    plt.fill_between([a, a, xmax, xmax], p, y2=ymin, color=c0, alpha=alph)

# Add special indicators for the landmarks with outcome 1.
ages0 = [CB[l]["Age"].value for l in CB.Ls[1]]
priors0 = [CB[l]["Priors"].value for l in CB.Ls[1]]
plt.scatter(
    ages0, # Age
    priors0, # Priors
    c = c1,
    s = 15,
    marker = 'x'
)

# Visualize the forcing cones of the landmarks with outcome 0.
xd = np.array([xmin, xmax])
for a, p in zip(ages0, priors0):
    plt.fill_between([a, a, xmin, xmin], p, y2=ymax, color=c1, alpha=alph)

# Save the resulting plot, if enabled. 
if save:
    plt.savefig(f"{save}.pdf")

# Show the plot.
plt.show()
print("\nDone.\n")