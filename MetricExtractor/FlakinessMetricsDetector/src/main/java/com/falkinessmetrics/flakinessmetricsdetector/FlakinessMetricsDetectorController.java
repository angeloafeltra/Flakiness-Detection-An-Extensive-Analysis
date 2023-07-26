package com.falkinessmetrics.flakinessmetricsdetector;

import computation.RunFlakinessMetricsDetection;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import java.io.File;

@RestController
 public class FlakinessMetricsDetectorController {


    @GetMapping(path = "/getFlakinessMetrics")
    public boolean getFlakinessMetrics(@RequestParam String repositoryName){

        System.out.println("Richiesta ricevuta Estrazione Metriche ricevuta");

        boolean result=true;
        String metricRepository = "../spazioCondiviso/MetricsDetector/"+repositoryName; //Da Utilizzare se non si passa per docker
        //String metricRepository = "./spazioCondiviso/MetricsDetector/"+repositoryName;
        File file = new File(metricRepository);
        System.out.println("Instanzio il metrics detector");
        RunFlakinessMetricsDetection detector=new RunFlakinessMetricsDetection();

        if (!file.exists()) {
            System.out.println("Estraggo le Metriche");
            result = detector.getMetrics(repositoryName);
        }else{
            System.out.println("Metriche gia estratte");
        }

        return result;
    }


    @GetMapping(path = "/testConnection")
    public String listenMessage(){
        return "Messaggio ricevuto e ricambiato";
    }


}
