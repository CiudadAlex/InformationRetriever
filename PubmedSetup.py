"""
For processing the big XML Pubmed files we will use bigxml:
https://bigxml.rogdham.net/quickstart/
"""

from bigxml import Parser, xml_handle_element, xml_handle_text
from bigxml import BigXmlError
from bigxml import HandlerTypeHelper, Streamable, XMLElement, XMLElementAttributes, XMLText

@xml_handle_element("PubmedArticleSet", "PubmedArticle")
def handler_pubmed_article(node):
    yield '-----------------------'
    yield from node.iter_from(handler_doi, handler_article)

@xml_handle_element("PubmedData", "ArticleIdList", "ArticleId")
def handler_doi(node):

    if node.attributes["IdType"] == "doi":
        yield "DOI: " + node.text  # node content as a str

@xml_handle_element("MedlineCitation", "Article")
def handler_article(node):
    yield from node.iter_from(handler_title, handler_abstract)

@xml_handle_element("ArticleTitle")
def handler_title(node):
    yield "Title: " + node.text  # node content as a str

@xml_handle_element("Abstract")
def handler_abstract(node):
    yield "Abstract: " + node.text  # node content as a str


def generateSeparatedFilesWithXml(file_path):

    with open(file_path, "rb") as f:
        count = 0
        for item in Parser(f).iter_from(handler_pubmed_article):

            if count == 100:
                break
            print(item)
            count = count + 1

generateSeparatedFilesWithXml("C:/Alex/Dev/data_corpus/InformationRetrieval/pubmed22n1513.xml")

