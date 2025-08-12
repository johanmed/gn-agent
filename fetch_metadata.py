from SPARQLWrapper import SPARQLWrapper, JSON

sparql=SPARQLWrapper(
    
    )
    
sparql.setReturnFormat(JSON)

sparql.setQuery(
    """
    
    """
    )
    
try:
    ret=sparql.queryAndConvert()
    
    output=ret["results"]["bindings"]:
        
except Exception as e:
    print(e)
    
