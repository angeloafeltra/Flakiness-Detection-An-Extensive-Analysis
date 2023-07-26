import pandas
from tqdm import trange
import sys
import requests

from Cloner import Cloner
from DatasetGenerator import DatasetGenerator

if __name__ == "__main__":

    if len(sys.argv)<3:
        print("Specificare csv e nome dataset da creare nella chiamata del main")
        exit()

    path_csv=sys.argv[1]
    name_dataset=sys.argv[2]

    repositories={
        "Nome": [],
        "URL": [],
        "SHA": [],
        "Lista_TF": [],
        "Lista_TNF": []
    }

    df=pandas.read_csv(path_csv,delimiter=';')

    for url,sha in zip(df['Project URL'],df['SHA Detected']):
        repoName=url.split('/')[-1]
        if not sha in repositories['SHA']:
            repositories['Nome'].append(repoName)
            repositories['URL'].append(url)
            repositories['SHA'].append(sha)
            repositories['Lista_TF'].append([])
            repositories['Lista_TNF'].append([])

    for index,row in df.iterrows():
        sha=row['SHA Detected']
        index_sha=repositories['SHA'].index(sha)
        listTF=repositories['Lista_TF'][index_sha]
        listTNF=repositories['Lista_TNF'][index_sha]

        test=row[df.columns[2]].replace('#','.')

        if 'IsFlaky' in df.columns:

            if row['IsFlaky']==0:
                listTNF.append(test)
            else:
                listTF.append(test)
        else:
            listTF.append(test)


    #Itero su ogni repository presente nel dataframe
    cloner=Cloner()
    datasetGeneretor=DatasetGenerator()
    datasetGeneretor.createDataset(name_dataset)
    progressbar=trange(len(repositories['Nome'])-1,desc='Cloning..',leave=True)
    for repository,url,sha,listTF,listTNF,_ in zip(repositories['Nome'],
                                                    repositories['URL'],
                                                    repositories['SHA'],
                                                    repositories['Lista_TF'],
                                                    repositories['Lista_TNF'],
                                                    progressbar):

        progressbar.set_description("Cloning {}_{}".format(repository,sha))
        progressbar.refresh()
        cloner.clone_repository(repository,url,sha)
        progressbar.set_description("Metrics Extract {}_{}".format(repository,sha))
        progressbar.refresh()
        PARAMS = {'repositoryName':'{}_{}'.format(repository,sha)}
        try:
            r=requests.get("http://flakiness_metrics_detector:8080/getFlakinessMetrics",params=PARAMS)
            progressbar.set_description("Generate Dataset {}_{}".format(repository,sha))
            progressbar.refresh()
            datasetGeneretor.addRepositoryToDataset('{}_{}'.format(repository,sha),listTF,listTNF)
        except Exception as e:
            progressbar.set_description("Errore di comunicazione col metricsDetector")
            progressbar.refresh()
            continue






