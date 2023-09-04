import metrics

def main():
    file_path = r"C:\Users\User\Downloads\example.csv"
    participant_1 = metrics.metrics(file_path)
    result = participant_1.getJson()
    print(result)
    
if __name__ == "__main__":
    main()
