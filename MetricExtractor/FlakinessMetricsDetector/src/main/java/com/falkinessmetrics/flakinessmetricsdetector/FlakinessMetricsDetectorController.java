package com.falkinessmetrics.flakinessmetricsdetector;

import computation.RunFlakinessMetricsDetection;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
 public class FlakinessMetricsDetectorController {

    private RunFlakinessMetricsDetection detector=new RunFlakinessMetricsDetection();
    @GetMapping(path = "/getFlakinessMetrics")
    public boolean getFlakinessMetrics(@RequestParam String repositoryName){
        boolean result=detector.getMetrics(repositoryName);
        return result;
    }


    @GetMapping(path = "/testConnection")
    public String listenMessage(){
        return "Messaggio ricevuto e ricambiato";

    }


}
