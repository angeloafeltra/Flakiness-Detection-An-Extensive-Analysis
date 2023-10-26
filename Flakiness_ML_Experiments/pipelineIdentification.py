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


    pipelineManager=PipelineManager()
    str=pipelineManager.getListPipelineName()

    print("Lista Pipeline:\n"+str)

    listPipeline=list()
    txt=""
    while (txt!='0') and (txt!='1'):
        txt=input("Specifica la pipeline da valutare indicando il nome (0 per terminare, 1 per valutarle tutte):")
        if txt!='0' and txt!='1':
            listPipeline.append(txt)
        if txt=='1':
            listPipeline.clear()
            for pipeline in str.split('\n'):
                listPipeline.append(pipeline)

    if len(listPipeline)>0:
        listPipeline=pipelineManager.getListPipeline(listPipeline)

        datasetManager=DatasetManager(datasetName)
        datasetManager.dataset_split(0.2,0.2)
        mlflowExperimentName='{}_Flakiness'.format(datasetName)
        experimentMenager=ExperimentManager(listPipeline,Custom_Tuning(),mlflowExperimentName)
        cv_Validator=CV_Validator()
        cpfp_Validator=CPFP_Validator()
        wpfp_Validator=WPFP_Validator()


        bestExperiment=experimentMenager.runExperiments(datasetManager.getTrainSet(),
                                                        datasetManager.getValidationSet(),
                                                        datasetManager.getTestSet(),
                                                        print_experiment=False,
                                                        log_mlflow=logMlflow)
        print("Miglior Pipeline")
        print(bestExperiment.toString())
        print()

        print("Cross-Validation")
        cv_Validator.validation(datasetManager.getFold(10),
                                bestExperiment.getPipeline(),
                                log_mlflow=logMlflow)
        print()
        print("CPFP-Validation")
        cpfp_Validator.validation(datasetManager.getListRepositorySet(),
                                  bestExperiment.getPipeline(),
                                  mlflowExperimentName,
                                  log_mlflow=logMlflow)
        print()
        print("WPFP-Validation")
        wpfp_Validator.validation(datasetManager.getListRepositorySet(),
                                  bestExperiment.getPipeline(),
                                  mlflowExperimentName,
                                  log_mlflow=logMlflow)



