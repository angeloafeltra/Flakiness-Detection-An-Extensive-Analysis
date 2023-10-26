import sys
import warnings

from management.PipelineManager import PipelineManager
from management.DatasetManager import DatasetManager
from management.ExperimentManager import ExperimentManager
from Custom_Tuning import Custom_Tuning
from validator.CV_Validator import CV_Validator
from validator.CPFP_Validator import CPFP_Validator
from validator.WPFP_Validator import WPFP_Validator


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    datasetName=sys.argv[1]
    logMlflow=True

    type_experiment=['CPBF_Experiment',
                     'CPLMC_Experiment',
                     'CPLMR_Experiment',
                     'CPTCA_Experiment',
                     'CPCTCA_Experiment',
                     'CPTrAda_Experiment']

    for experiment in type_experiment:
        print(experiment)


    listExperiment=list()
    txt=""
    while (txt!='0') and (txt!='1'):
        txt=input("Specifica l'esperimento da eseguire (0 per terminare, 1 per valutarle tutte):")
        if txt!='0' and txt!='1':
            listExperiment.append(txt)
        if txt=='1':
            listExperiment=type_experiment

    if len(listExperiment)>0:
        mlflowExperimentName='{}_Flakiness'.format(datasetName)
        datasetManager=DatasetManager(datasetName)
        cpfp_Validator=CPFP_Validator()
        list_repository=datasetManager.getListRepositorySet()

        for repository in list_repository:
            if cpfp_Validator.filtro(repository[2]): print(repository[0].split('_')[0])

        listRep=list()
        txt=""
        while (txt!='0') and (txt!='1'):
            txt=input("Specifica la repositoryi da validare 0 per terminare, 1 per validarle tutte):")
            if txt!='0' and txt!='1':
                listRep.append(txt)
            if txt=='1':
                listRep=None

        cpfp_Validator.validation(listExperiment,list_repository,None,mlflowExperimentName,True,listRep)



