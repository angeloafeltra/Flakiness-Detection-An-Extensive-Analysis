import requests


if __name__ == "__main__":

    r=requests.get("http://flakiness_metrics_detector:8080/testConnectio")
    print(r.text)






