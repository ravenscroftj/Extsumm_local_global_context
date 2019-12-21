"""CLI tool for generating summaries of scientific papers"""

import click
import logging
import spacy
import ujson as json
import os
import torch

from lxml import etree
from typing import List
from models import *

logger = logging.getLogger("xiao-summarizer")

@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if verbose:
        logger.info("Enabling verbose output")

@cli.command()
@click.argument("paper_files", type=click.Path(exists=True), nargs=-1, required=True)
@click.argument("output_folder", type=click.Path())
@click.option("-m","--spacy-model", type=str, default="en")
def convert(paper_files, output_folder, spacy_model):
    """Convert DocoXML Standard to JSON compatible with Xiao model"""
    
    if not os.path.exists(output_folder):
        logger.info(f"Attempt to create output folder {output_folder}")
        os.makedirs(output_folder)
    
    
    logger.info(f"Convert {len(paper_files)} files and store in {output_folder}")
    
    logger.info(f"Initialise spacy with {spacy_model} language model")
    nlp = spacy.load(spacy_model)
    
    for file in paper_files:
        
        basename = os.path.basename(file)
        fileid = os.path.splitext(basename)[0]
        
        outfile = os.path.join(output_folder,  fileid + ".json")
        
        if os.path.exists(outfile):
            logger.info(f"Skipping {file} since summary {outfile} already exists")
            continue
        
        doc = etree.parse(file)
        
        texts = []
        sections = []
        if doc.find(".//section") != None:
            logger.debug("Parsing DoCoXML")
            for sec in doc.findall('.//section'):
                section_name = sec.find("./*[@class='DoCO:SectionTitle']").text
                text = " ".join([text.strip() for text in sec.itertext()])
                texts.append(text)
                sections.append(section_name)
                
        elif doc.find(".//sec[@sec-type]") != None:
            logger.debug("Parsing PMC format")
            
            for sec in doc.findall('.//sec[@sec-type]'):
                if sec.find("./title") != None:
                    section_name = sec.find("./title").text
                else:
                    section_name = ""
                    
                text = ". ".join([text.strip() for text in sec.itertext()])
                texts.append(text)
                sections.append(section_name)
        
        if len(texts) < 1:
            logger.warn(f"No text found in {file} so skipping...")
            continue
        
        out = segment_documents(nlp, texts, sections)
        out['id'] = fileid

        
        with open(outfile, "w") as f:
            json.dump(out, f, indent=2)
        
def segment_documents(nlp: spacy.language.Language, texts: List[str], section_titles: List[str]):
    """Segment XML sentences and generate a list of sentences"""
         
    docs = nlp.pipe(texts)
    
    sentence_id = 1
    
    out = {'inputs':[], 'section_names': [], 'section_lengths': []} 
    
    # iterate over sections in paper
    for section_name, doc in zip(section_titles, docs):
        
        sentcount = 0
        
        # iterate over sentences in section
        for sent in doc.sents:
            tokens = [tok.text for tok in sent]
            input_doc = {
                "text": sent.text,
                "tokens": tokens,
                "sentence_id": sentence_id,
                "word_count": len(tokens)
            }
            
            out['inputs'].append(input_doc)
            
            sentence_id += 1
            sentcount += 1
            
        # end iterating over sentences
        out['section_lengths'].append(sentcount)
        out['section_names'].append(section_name)
    
    return out
        
    
    

@cli.command()
@click.argument("paper_files", type=click.Path(exists=True), nargs=-1, required=True)
@click.argument("output_folder", type=click.Path(dir_okay=True))
@click.option("--model-type", type=str, default="concat")
@click.option("--model-state-file", type=click.Path(exists=True, file_okay=True), default="pretrained_models/Pubmed/concat")
@click.option("--batch_size", type=int, default=32)
@click.option("--hidden_dim", type=int, default=300)
@click.option("--embedding_dim", type=int, default=300)
@click.option("--mlp_size", type=int, default=100)
@click.option("--cell_type", type=str, default="gru")
@click.option("--length_limit", type=int, default=290)
@click.option("--use_section_info", type=bool, default=True)
@click.option("--vocab_size", type=int, default=50000)
@click.option("--vocab_dataset", type=str, default="pubmed")
@click.option("--glove_dir", type=click.Path(dir_okay=True, exists=True), default=".")
def summarise(paper_files, output_folder, model_type, model_state_file, batch_size, hidden_dim, embedding_dim, mlp_size, cell_type, length_limit, use_section_info, vocab_dataset, vocab_size, glove_dir ):
    """Generate summaries for scientific documents"""
    
    from utils import getEmbeddingMatrix
    
    if os.path.exists(f"vocabulary_{vocab_dataset}.json"):
        logger.info(f"Loading vocabulary from vocabulary_{vocab_dataset}.json")

        with open(f"vocabulary_{vocab_dataset}.json",'r') as f:
            w2v = json.load(f)
    
    else:
        import zipfile
        from utils import build_word2ind_zip
        
        logger.info("attempting to build vocab from dataset zip")
        with zipfile.ZipFile(f"{vocab_dataset}.zip", mode='r') as z:
            files = z.namelist()
            train_files = [file for file in files if file.startswith("pubmed/inputs/train") and file.endswith(".json")]
            w2v = word2index = build_word2ind_zip(z, train_files, vocab_size)
        
        with open("vocabulary_{vocab_dataset}.json",'w') as f:
            json.dump(w2v,f)
            
    # load vectors
    logger.info(f"Attempting to load glove vectors from {glove_dir}")
    embeddings = getEmbeddingMatrix(glove_dir, w2v, 300)
    
    #turn on torch
    if torch.cuda.is_available():
        logger.info("Turn on torch with CUDA")
        device = torch.device(type='cuda')
    else:
        logger.info("CUDA not available, turn on torch with CPU")
        device = torch.device(type='cpu')
        
    logger.info(f"Torch device={device}")
        
    # load model
    logger.info("Load model")
    state_dict = torch.load(model_state_file,  map_location=device)
    model = load_model(model_type, device, embedding_dim, hidden_dim, mlp_size, cell_type, state_dict)
    
    # generate list of files
    from pathlib import Path
    input_file_paths = [Path(file) for file in paper_files]
    
    logger.info(f"Prepare dataset of {len(input_file_paths)} to process...")
    
    from data import SummarizationDataset, SummarizationDataLoader
    
    sz = SummarizationDataset(w2v, embeddings, embedding_dim, input_dir=input_file_paths)
    loader = SummarizationDataLoader(sz, batch_size=batch_size)
    
    for i, data in enumerate(loader):
        data_batch = data
        
        document = data_batch['document']
        label = data_batch['labels']
        input_length = data_batch['input_length']
        indicators = data_batch['indicators']
        padded_lengths = data_batch['padded_lengths']
        ids = data_batch['id']
        filenames = [Path(filename) for filename in data_batch['filenames']]
        
        total_data = torch.sum(input_length)
        end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
        begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)

        document = document.to(device)
        if label != None:
            label = label.to(device)
        input_length = input_length.to(device)
        indicators = indicators.to(device)
        end = end.to(device)
        begin= begin.to(device)
            
        out= model(document,input_length,indicators,begin,end,device)
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(out).data
    
        summaryfiles,all_ids= model.predict(scores, ids, input_length, length_limit, filenames, Path(output_folder))
        logger.info(f"Generated summary for {all_ids} at {summaryfiles}")
    
    
def load_model(model_type, device, embedding_dim, hidden_dim, mlp_size, cell_type, state_dict):
    """Utility method for loading model type"""
    
    
    if model_type == "concat":
        model = Concatenation(embedding_dim, hidden_dim, mlp_size, cell_type)
    elif model_type == 'bsl1':
        model = Bsl1(embedding_dim, hidden_dim, mlp_size, cell_type)
    elif model_type == 'bsl2':
        model = Bsl2(embedding_dim, hidden_dim, mlp_size, cell_type)
    elif model_type == 'bsl3':
        model = Bsl3(embedding_dim, hidden_dim, mlp_size, cell_type)
    elif model_type == 'attentive_context':
        model = Attentive_context(embedding_dim, hidden_dim, mlp_size, cell_type)
    elif model_type == 'cl':
        model = ChengAndLapataSentenceExtractor(embedding_dim, hidden_dim, cell_type)
    elif model_type == 'sr':
        model = SummaRunnerSentenceExtractor(embedding_dim, hidden_dim, cell_type)
        
    model = model.to(device)
    model.load_state_dict(state_dict)

    return model

if __name__ == "__main__":
    cli()