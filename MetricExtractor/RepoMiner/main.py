from time import sleep

import pandas
from tqdm import tqdm
import sys
import requests

from Cloner import Cloner

if __name__ == "__main__":

    repositories={
        "Nome": [],
        "URL": [],
        "SHA": [],
        "Lista_TF": [],
        "Lista_TNF": []
    }

    #df=pandas.read_csv(sys.argv[1])
    df=pandas.read_csv('/Users/angeloafeltra/Documents/GitHub/Flakiness-Detection-An-Extensive-Analysis/ListaTestFlaky/FlakeFlagger.csv',delimiter=';')

    print(df.info())#Da Cancellare

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

        if 'IsFlaky' in df.columns:
            if row['IsFlaky']==0:
                listTNF.append(row[df.columns[2]])
            else:
                listTF.append(row[df.columns[2]])
        else:
            listTF.append(row[df.columns[2]])

    '''
    counterTF=0
    counterTNF=0
    for repo,SHA,listTF,listTNF in zip(repositories['Nome'],repositories['SHA'],repositories['Lista_TF'],repositories['Lista_TNF']):
        counterTF=counterTF+len(listTF)
        counterTNF=counterTNF+len(listTNF)
        print('{} : {} : TF={} : TNF={}'.format(repo,SHA,len(listTF),len(listTNF)))

    print("Numero Test Flaky: {}\nNumero Test Non Flaky:{}".format(counterTF,counterTNF))
    '''
    #Itero su ogni repository presente nel dataframe
    cloner=Cloner()
    i=0
    for repository,url,sha,listTF,listTNF,_ in zip(repositories['Nome'],
                                                 repositories['URL'],
                                                 repositories['SHA'],
                                                 repositories['Lista_TF'],
                                                 repositories['Lista_TNF'],
                                                 tqdm(range(len(repositories['Nome'])-1),desc='Cloning..')):
        if i<2:
            #cloner.clone_repository(repository,url,sha)
            PARAMS = {'repositoryName':'{}_{}'.format(repository,sha)}
            r=requests.get("http://localhost:8080/getFlakinessMetrics",params=PARAMS)
            result=r.text
            print(result)
            i=i+1

        sleep(0.1)



