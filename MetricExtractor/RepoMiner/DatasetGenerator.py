import os.path

import pandas as pd

class DatasetGenerator:

    def __init__(self): pass

    def createDataset(self,datasetName):
        self.datasetName=datasetName;
        self.pathDataset=os.path.join("../spazioCondiviso/Dataset",'{}.csv'.format(datasetName)) #Da utilizzare se non si passa per docker
        #self.pathDataset=os.path.join("./spazioCondiviso/Dataset",
                                #'{}.csv'.format(datasetName))
        if not os.path.exists(self.pathDataset):
            df = pd.DataFrame(columns=['nameProject', 'testCase', 'tloc', 'tmcCabe',
                                    'assertionDensity', 'assertionRoulette', 'mysteryGuest', 'eagerTest',
                                    'sensitiveEquality', 'resourceOptimism', 'conditionalTestLogic', 'fireAndForget',
                                    'testRunWar', 'loc', 'lcom2', 'lcom5',
                                    'cbo', 'wmc', 'rfc', 'mpc',
                                    'halsteadVocabulary', 'halsteadLength', 'halsteadVolume',
                                    'classDataShouldBePrivate', 'complexClass', 'functionalDecomposition'
                                    'godClass','spaghettiCode','isFlaky'])

            df.to_csv(self.pathDataset,index=False)


    def addRepositoryToDataset(self,repository,listTestFlaky,listTestNonFlaky):
        df=pd.read_csv(self.pathDataset)
        if not self.repositoryExistInDataset(repository):
            df_repository=self.labellingRepositoryCSV(repository,listTestFlaky,listTestNonFlaky)
            df = pd.concat([df, df_repository])
            df.reset_index(drop=True)
            df.to_csv(self.pathDataset,index=False)


    def repositoryExistInDataset(self,repository):
        df=pd.read_csv(self.pathDataset)
        list_repo=df['nameProject'].unique()
        if repository in list_repo: return True

        return False

    def labellingRepositoryCSV(self,repsitory,listTestFlaky,listTestNonFlaky):
        pathDataset=os.path.join("../spazioCondiviso/MetricsDetector/",'{}'.format(repsitory)) #Da utilizzare se non si passa per docker
        #pathDataset=os.path.join("./spazioCondiviso/MetricsDetector/",
                                #'{}'.format(repsitory))
        df=pd.read_csv(pathDataset)
        if len(listTestNonFlaky)==0:
            df['isFlaky']=0
        else:
            df['isFlaky']=-1

        for testFlaky in listTestFlaky:
            df.loc[df['testCase'].str.contains(testFlaky, case=False), 'isFlaky'] = 1
        for testNonFlaky in listTestNonFlaky:
            df.loc[df['testCase'].str.contains(testNonFlaky, case=False), 'isFlaky'] = 0

        df = df[df.isFlaky != -1]
        df.reset_index(drop=True)
        return df