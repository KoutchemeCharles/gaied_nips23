import pandas as pd 
from markdownify import MarkdownConverter

class TableConverter(MarkdownConverter):
    """
    Create a custom MarkdownConverter that adds two newlines after an image
    """
    def convert_table(self, el, text, convert_as_inline):
        table_str = str(el)
        # table_str = re.sub(r'<.*?>', lambda g: g.group(0).upper(), table_str)
        # If the table is empty, it will not work, by default, we remove that table 
        # TODO: improve here what we do 
        try:
            tdf = pd.read_html(table_str)[0]
            tdf = tdf.fillna("")
            tdf = tdf.rename( columns={'Unnamed: 0':''})
            return tdf.to_string() # alternative tdf.to_csv(header=None)
        except:
            return ""

