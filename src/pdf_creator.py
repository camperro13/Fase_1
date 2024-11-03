from bs4 import BeautifulSoup

# Specify the path to your HTML report
html_file = '/Users/juancarloscamperovilla/Documents/GitHub/MLOps/Residencial_build/Fase_1/report_testing.html'
txt_file = '/Users/juancarloscamperovilla/Documents/GitHub/MLOps/Residencial_build/Fase_1/report_testing.txt'

def html_to_txt(html_file, txt_file):
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Extract text from the HTML
    text = soup.get_text(separator='\n')  # Use \n to separate different tags

    # Write the text to a .txt file
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(text)

    print(f"Text extracted and saved to {txt_file}")

# Example usage
html_to_txt(html_file, txt_file)