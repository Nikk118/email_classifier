import numpy as np

class mulitnomialNB():
        def fit(self,x,y):
                self.classes=np.unique(y)
                self.class_log_prior={}
                self.feature_log_prior={}
                self.alpha=1

                total_docs=len(y)
                for c in self.classes:
                        x_c=x[y==c]
                        total_c_docs=x_c.shape[0]
                        self.class_log_prior[c]=np.log(total_c_docs/total_docs)

                        words_count=x_c.sum(axis=0)+ self.alpha
                        total_words=words_count.sum()

                        self.feature_log_prior[c]=np.log(words_count/total_words)
        def predict(self,x):
                results=[]
                for i in range(x.shape[0]):
                        sample=x[i]
                        class_score={}
                        for c in self.classes:
                                log_prior=self.class_log_prior[c]
                                log_liklihood=sample@ self.feature_log_prior[c].T
                                score=log_prior+log_liklihood
                                class_score[c]=score
                        predict_class=max(class_score,key=class_score.get)
                        results.append(predict_class)
                return np.array(results)
