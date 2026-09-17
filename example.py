import random

phenotypes = [
    "Hypertension",
    "Diabetes",
    "Asthma",
    "Obesity",
    "Depression",
    "Anxiety",
    "Heart Disease"
]

drugs = [
    "Lisinopril",
    "Metformin",
    "Albuterol",
    "Orlistat",
    "Sertraline",
    "Buspirone",
    "Aspirin"
]

def gen_phenotypes():
    return [phenotype if random.randint(0, 100) == 0 else None
                for phenotype in phenotypes]

def gen_genotype():
    return [f"{j}{random.randint(0, 2)}" for j in range(5)]


def gen_drugs(patient_phenotypes):
    return [
        drug if phenotype is not None and random.randint(0, 100) == 0 else None
        for drug, phenotype in zip(drugs, patient_phenotypes)
    ]


def prs(patient, phenotypes=phenotypes):
    # Placeholder PRS
    return [
        sum(
            hash(f"{patient['id']}-{genotype}-{phenotype}")
            for genotype in patient["genotype"]
        ) % 10000 / 10000
        for phenotype in phenotypes
    ]

def gen_patients(n=1000):
    patients = []

    for i in range(n):
        patient = {
            "id": i,
            "name": f"Patient {i}",
            "genotype": gen_genotype(),
            "phenotypes": gen_phenotypes()
        }

        patient["drugs"] = gen_drugs(patient["phenotypes"])
        patient["prs"] = prs(patient, phenotypes)
        patients.append(patient)

    return patients

if __name__ == "__main__":
    # gen_patients_phenotypes.csv
    # Phenotypes are columns, patient ID rows, cells are presence
    with open("gen_patients_phenotypes.csv", "w") as f:
        f.write("id," + ",".join(phenotypes) + "\n")
        for patient in gen_patients():
            row = [str(patient["id"])] + [str(int(phenotype is not None)) for phenotype in patient["phenotypes"]]
            f.write(",".join(row) + "\n")

    # gen_patients_prs.csv
    # Phenotypes are columns, patient ID rows, cells are PRS values
    with open("gen_patients_prs.csv", "w") as f:
        f.write("id," + ",".join(phenotypes) + "\n")

        for patient in gen_patients():
            row = [str(patient["id"])]

            for prs_value in patient["prs"]:
                row.append(f"{prs_value:.4f}")

            f.write(",".join(row) + "\n")


    # gen_patients_drugs.csv
    # Drugs are columns, patient ID rows, cells are presence

    with open("gen_patients_drugs.csv", "w") as f:
        f.write("id," + ",".join(drugs) + "\n")

        for patient in gen_patients():
            row = [str(patient["id"])] + [
                str(int(drug is not None))
                for drug in patient["drugs"]
            ]

            f.write(",".join(row) + "\n")
        