from FUNCS import get_file


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
filename = 'teleCust1000t.csv'
df = get_file(url, filename)

print(df.head())

