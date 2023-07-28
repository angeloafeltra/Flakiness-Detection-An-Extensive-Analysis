# Flakiness Detection: An Extensive Analysis

I casi di test per un sistema software assicurano che le modifiche effettuate al codice non vadano ad influire negativamente sulle funzionalità già esistenti, tant’è vero che quando gli sviluppatori apportano modifiche al loro codice, eseguono il cosiddetto test di regressione per rilevare se le modifiche effettuate introducono eventuali bug all’interno del sistema.
Tuttavia, alcuni test potrebbero essere flaky, ovvero non deterministici è assumere sia un comportamento di pass che di failure quando vengono eseguiti sullo stesso codice. 
Inizialmente la flakiness è stata affrontata rieseguendo più volte il caso di test e osservando il proprio comportamento. Tale approccio è dispendioso in termini di tempo e costoso dal punto di vista computazionale, pertanto ad oggi si è deciso di utilizzare il machine learning per prevederla.
Diversi studi sono stati condotti su diversi dataset, seguendo delle metodologie ad hoc che puntavano a verificare più la fattibilità dell’utilizzo del machine learning che la reale utilità pratica utilizzando un’approccio in-vitro.
Con tale lavoro cerchiamo di fare l’esatto opposto, ovvero utilizzare un approccio in-vivo per mostrare la reale utilità pratica del machine learning verso la flakiness.



Per poter replicare il seguente lavoro bisogna eseguire i seguenti passi:

## Generazione Dataset
Per replicare i nostri dataset utilizzare la lista di test flaky presenti nella cartella ListaTestFlaky per poter generare i relativi dataset.

### Metodo 1 (non richiede l'utilizzo di Docker)
1. Clonare la repository git
2. Eseguire il main ./MetricExtractor/FlakinessMetricsDetector/src/main/java/com.flakinessmetrics.flakinessmetricsdetector/FlakinessMetricsDetectionApplication.java
3. Spostarsi nella cartella ./MetricExtractor/RepoMiner
4. Eseguire il comando: python main.py [path-csv-listaTestFlaky] [nome_dataset_da_generare]
5. Il dataset generato è presente nella cartella ./MetricExtractor/SpazioCondiviso/Dataset

### Metodo 2 (richiee l'utilizzo di Docker)
1. Clonare la repository git
2. Spostarsi nella cartella ./MetricExtractor
3. Eseguire il comando: docker-compose -d
4. Eseguire il comando: docker-compose cp [path-csv-listaTestFlaky] repo_miner:/RepoMiner/[nome_csv_listaTestFlaky]
5. Eseguire il comando: docker-compose exec repo_miner bash
6. Eseguire il comando: python main.py [nome_csv_listaTestFlaky] [nome_dataset_da_generare]
7. Al termine della generazione del dataset eseguire il comando: docker-compose cp repo_miner:/spazioCondiviso/Dataset/[nome_dataset_da_generare].csv [path_local_folder]

E consigliabile utilizzare Docker in quanto con windows si potrebbero avere degli errori durante la fase di clone, data la creazione di cartelle con un nome troppo lungo.



## Dataset Analysis
Dopo la generazione dei vari dataset utilizzando il file Data_Analysis.Rmd è possibile eseguire una fase di data cleaning per miglioare la qualità dei dataset nella quale verrano eseguiti i seguenti passaggi:
1. Rimozione repository senza test flaky
2. Mantenimento di un solo commit per repository (il commit con più test flaky)
3. Rimozione test setup e teardown
4. Rimozione duplicati
5. Rimozione rumore (test non flaky con la stessa metrica dei test flaky)

Tuttavia per eseguire tale analisi all'interno di tale file Data_Analysis.Rmd deve essere specificato il path ed il nome del dataset da analizzare

## Machine Learning Experiments 

## With-in Evaluated

## Cross-project Evaluated
