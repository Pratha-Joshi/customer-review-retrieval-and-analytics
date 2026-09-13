from __future__ import annotations
import pandas as pd
class AnalyticsService:
    def __init__(self, documents):
        self.df=pd.DataFrame([{**d['metadata'], 'id':d['id']} for d in documents])
    def filter_ids(self, product_id=None, rating_min=None, rating_max=None):
        frame=self.df
        if product_id: frame=frame[frame.product_id.astype(str)==str(product_id)]
        if rating_min is not None: frame=frame[frame.rating>=rating_min]
        if rating_max is not None: frame=frame[frame.rating<=rating_max]
        return set(frame.id)
    def summary(self, product_id=None):
        frame=self.df[self.df.product_id.astype(str)==str(product_id)] if product_id else self.df
        if frame.empty: return {"review_count":0, "message":"No matching reviews."}
        return {"product_id": product_id, "review_count":len(frame), "average_rating":float(frame.rating.mean()),
          "rating_distribution":{str(k):int(v) for k,v in frame.rating.value_counts().sort_index().items()},
          "helpfulness":{"mean":float(frame.helpfulness_ratio.mean()), "median":float(frame.helpfulness_ratio.median())},
          "review_volume_by_year":{str(k):int(v) for k,v in frame.review_time.astype(str).str[:4].value_counts().sort_index().items() if k != 'None'}}
    def compare(self, product_a, product_b): return {"product_a":self.summary(product_a), "product_b":self.summary(product_b)}
