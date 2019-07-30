# tweet_classifier_plus_eli5

Goals for Part 1:
- Revisit AOC Twitter reply dataset
- Make a text classifier with SciKit basics (looks like Linear for now)
- Use ELI5's tools to highlight text used in the prediction

Goals for Part 2:
- Make a text classifier with FastText
- Use ELI5's TextExplainer to query the FastText model and highlight text used in the prediction

Goals for Part 3:
- API for new Tweets
- Inject predictions into Twitter frontend

```
virtualenv -p python3 .env
source .env/bin/activate

pip3 install scikit-learn eli5 numpy xgboost
```
