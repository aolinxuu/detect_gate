import csv
import os
import PyPDF2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file located at pdf_path"""

    # Open PDF in binary read mode
    with open(pdf_path, 'rb') as pdf_file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(pdf_file)

        text = ''
        # Iterate through each page in the PDF
        for page in reader.pages:
            # Extract text from page and add to text string
            text += page.extract_text()

    return text


def write_text_to_csv(text, csv_path):
    """Write a string of text to a CSV file located at csv_path"""

    # Open CSV file in write mode
    with open(csv_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        writer = csv.writer(csv_file)
        # Write text string to CSV in a single row
        writer.writerow([text])


def pdf_to_csv(pdf_path, csv_path):
    """Convert a PDF to CSV by extracting all text from the PDF and writing it to a CSV file"""

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    # Write text to CSV
    write_text_to_csv(text, csv_path)


if __name__ == "__main__":
    # Get PDF file path from environment variable
    pdf_path = os.getenv("report_path")

    # Get directory of PDF file
    dir_path = os.path.dirname(pdf_path)

    # Define CSV file path to save in the same directory as the PDF file
    csv_path = os.path.join(dir_path, 'output.csv')

    # Convert PDF to CSV
    pdf_to_csv(pdf_path, csv_path)
