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

        boolean result=true;
        String metricRepository = ".spazioCondiviso/MetricsDetector/"+repositoryName;
        File file = new File(metricRepository);
        RunFlakinessMetricsDetection detector=new RunFlakinessMetricsDetection();
        if (!file.exists())
            result = detector.getMetrics(repositoryName);

        return result;
    }


    @GetMapping(path = "/testConnection")
    public String listenMessage(){
        return "Messaggio ricevuto e ricambiato";

    }


}
