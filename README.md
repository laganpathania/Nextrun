# NextRun: experiment planner and lab simulator

Stop guessing your next experiment. NextRun suggests the next batch of experiments with Bayesian optimization, and it learns from every result you add. It also includes a simulated lab, so you can watch a whole campaign run with instruments, random failures and a maintenance window.

**Try it live:** https://laganpathania.com/autolab/

## What it does

| Tab | What you can do |
|---|---|
| About this project | The challenge, the approach and the limits of the tool |
| Demo | Run a simulated campaign with Bayesian optimization or random search, then inspect the learning curve, the instrument timeline, per-round utilization and the audit trail |
| Try with your data | Define your numeric variables and one goal, optionally upload past experiments (Excel or CSV), and get the next batch to run, plus a model check |

## How it chooses experiments

- Early on, with too little data to fit a model, it spreads experiments across your ranges.
- Once there are enough results, it fits a Gaussian process and picks each batch by Expected Improvement. A constant-liar step stops a batch from repeating the same point.
- Every step is written to a hash-chained audit log, and the app checks that the chain is intact.

## Result on the built-in demo assay

After 48 experiments, averaged over 10 repeated runs, Bayesian optimization reached a best signal-to-background of **7.7** against **3.8** for random search. The two methods are indistinguishable for roughly the first 35 experiments.

This assay is a function written for this project. The result shows that the method works, not that it works on a real assay. Single runs vary widely, so quote the average.

## Run it on your computer

Use Python 3.11 to 3.13.

```
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

```
app.py          the Streamlit interface
config.yaml     every number lives here: factors, constraints, instruments, synthetic assay
labtwin/        planner, simulated devices, orchestrator, audit log, data import and export
requirements.txt
```

## Limits

- One response at a time, with continuous or integer variables and simple minimum and maximum limits. Categorical variables are not supported.
- No tool can say in advance how many experiments you will need. Watch the best-so-far curve and decide against your budget.
- The simulated lab is a model, not validated against real instruments. The OPC UA interface of the full project is not part of the browser version.
- A decision aid, not validated software. You remain responsible for what is run in the lab.

## Privacy

The live version runs in your browser. Your files are not uploaded to a server.

## Author

Lagan Pathania, lab automation architect. [laganpathania.com](https://laganpathania.com)
