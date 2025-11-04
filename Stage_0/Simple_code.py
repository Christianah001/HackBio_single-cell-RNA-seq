"""
HackBio Internship - Stage 0 Python Task
Team: Aspartic

# Task: Write a simple Python script for printing the names, slack username, country, 1 hobby, affiliations of people on your team and the DNA sequence of the genes they love most.
# Author: Funmilayo Christianah Ligali
# GitHub: https://github.com/Christianah001
# LinkedIn: https://www.linkedin.com/in/funmilayo-ligali-9746a2184/
"""

# Team information stored in a list of dictionaries
# Each dictionary represents one member
team_aspartic_info = [
        {"Name": "Funmilayo", "Slack": "Funmilayo Ligali", "Country": "Nigeria", "Hobby": "Cooking",
     "Affiliation": "Biochemistry", "Favorite Gene": "TP53", "Sequence": "ATGGAGGAGCCGCAGTCAGAT"},
    {"Name": "Akinjide", "Slack": "Akinjide Anifowose", "Country": "Nigeria", "Hobby": "Coding",
     "Affiliation": "Biomedical Informatics", "Favorite Gene": "PTEN", "Sequence": "ATGACAGCCATCATCAAAGAG"},
    {"Name": "Fakhia Mubashir", "Slack": "Fakhia Mubashir", "Country": "Pakistan", "Hobby": "Reading",
     "Affiliation": "None", "Favorite Gene": "TRPV6", "Sequence": "ATCGTCGCCCGATACGATCCAGTA"},
    {"Name": "Niraj Pun Magar", "Slack": "Niraj Pun Magar", "Country": "Nepal", "Hobby": "UX design",
     "Affiliation": "Bioinformatics", "Favorite Gene": "APOE", "Sequence": "ATGCATGCGCGCATGC"},
    {"Name": "Himanshu Pundir", "Slack": "Himanshu Pundir", "Country": "India", "Hobby": "Editing",
     "Affiliation": "Bioinformatics", "Favorite Gene": "SIR1", "Sequence": "ATGTCTATAAAAGGAAAT"},
    {"Name": "Diksha Shetty", "Slack": "Diksha Shetty", "Country": "India", "Hobby": "Reading",
     "Affiliation": "Bioinformatics", "Favorite Gene": "MYC", "Sequence": "GGAGTTTATTCATAACGCGCTCTCCAAGTATACGTGGCAATGCGTT"},
    {"Name": "Sri Sathya Sandilya Garemilla", "Slack": "Garemilla Sandilya", "Country": "United States", "Hobby": "Coding",
     "Affiliation": "Bioinformatics and Molecular Biochemistry", "Favorite Gene": "NIL", "Sequence": "AACCGCATCTGCAGCGAGCATCTGAGAAGCCAAGACTGAGCCGGCGGCCGCGGCGCAGCGAACGAGCAGT"},
]

# To define a simple function to print member details
def print_member(member):
    """Prints formatted details of a team member."""
    print(f"{member['Name']:<30}{member['Slack']:<25}{member['Country']:<15}"
          f"{member['Hobby']:<20}{member['Affiliation']:<45}"
          f"{member['Favorite Gene']:<10}{member['Sequence']}")

# To print a title 
print("\nHackBio Team Aspartic Information:\n")

# To print header for clarity
print("\nHackBio Team Aspartic Information:\n")
print(f"{'Name':<30}{'Slack Username':<25}{'Country':<15}"
      f"{'Hobby':<20}{'Affiliation':<45}"
      f"{'Fav Gene':<10}{'Sequence'}")
print("-" * 170)

# Loop through team members and print each one
for member in team_aspartic_info:
    print_member(member)

# To print total team members
print(f"\nTotal_team_members: {len(team_aspartic_info)}")
