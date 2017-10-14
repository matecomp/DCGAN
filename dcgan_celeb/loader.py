# -*- encoding: utf-8 -*-

""" Aviso: este future print e division permite que o codigo funcione
    tanto em python3 quanto em python 2.x"""
from __future__ import print_function, division
from builtins import range

import os
import requests
import zipfile
import numpy as np
import pandas as pd
from scipy.misc import imread, imsave, imresize
from glob import glob
from tqdm import tqdm


# vamos baixar e preparar o dataset
def get_celeb(limit=None):
  if not os.path.exists('../large_files'):
    os.mkdir('../large_files')

  # vamos criar uma nova pasta com as imagens cortadas
  if not os.path.exists('../large_files/img_align_celeba-cropped'):

    # caso não tenhamos o dado original, temos que fazer o download
    if not os.path.exists('../large_files/img_align_celeba'):
      # vamos fazer o download e armazenar aqui
      if not os.path.exists('../large_files/img_align_celeba.zip'):
        print("Downloading img_align_celeba.zip...")
        download_file(
          '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
          '../large_files/img_align_celeba.zip'
        )

      # vamos descompactar, o que pode demorar uma eternidade...
      print("Extracting img_align_celeba.zip...")
      with zipfile.ZipFile('../large_files/img_align_celeba.zip') as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall('../large_files')


    # tendo já o aquivo original, não é necessário descompactar e nem baixar, vamos abrir as imagens
    filenames = glob("../large_files/img_align_celeba/*.jpg")
    N = len(filenames)
    print("Found %d files!" % N)


    # vamos criar uma nova pasta e armazenar as imagens preprocessadas nesta pasta
    os.mkdir('../large_files/img_align_celeba-cropped')
    print("Cropping images, please wait...")

    for i in range(N):
      crop_and_resave(filenames[i], '../large_files/img_align_celeba-cropped')
      if i % 1000 == 0:
        print("%d/%d" % (i, N))


  # vamos retornar os endereços das imagens cortadas, caso quisermos a imagem, podemos chamar a funcao files2images
  filenames = glob("../large_files/img_align_celeba-cropped/*.jpg")
  return filenames


# Cortaremos a imagem para capturar apenas o rosto, eliminando ambiente
def crop_and_resave(inputfile, outputdir):
  # teoricamente, nós podemos encontrar a face com alguma IA
  # mas seremos preguiçosos
  # vamos assumir que o rosto está sempre no centro da imagem
  im = imread(inputfile)
  height, width, color = im.shape
  edge_h = int( round( (height - 108) / 2.0 ) )
  edge_w = int( round( (width - 108) / 2.0 ) )

  cropped = im[edge_h:(edge_h + 108), edge_w:(edge_w + 108)]
  small = imresize(cropped, (64, 64))

  filename = inputfile.split('/')[-1]
  imsave("%s/%s" % (outputdir, filename), small)


# Escalar os valores de [0,255] para [-1,1]
def scale_image(im):
  return (im - 128.0)/127.

# Fara a leitura de cada aquivo de imagem e salvar numa lista
def files2images(filenames):
  return [scale_image(imread(fn)) for fn in filenames]





# funcoes para fazer download no google drive
### as funcoes abaixo foram feitas pelo Lazyprogrammer
def save_response_content(r, dest):
  # unfortunately content-length is not provided in header
  total_iters = 1409659 # in KB
  print("Note: units are in KB, e.g. KKB = MB")
  # because we are reading 1024 bytes at a time, hence
  # 1KB == 1 "unit" for tqdm
  with open(dest, 'wb') as f:
    for chunk in tqdm(
      r.iter_content(1024),
      total=total_iters,
      unit='KB',
      unit_scale=True):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)


def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None


def download_file(file_id, dest):
  drive_url = "https://docs.google.com/uc?export=download"
  session = requests.Session()
  response = session.get(drive_url, params={'id': file_id}, stream=True)
  token = get_confirm_token(response)

  if token:
    params = {'id': file_id, 'confirm': token}
    response = session.get(drive_url, params=params, stream=True)

  save_response_content(response, dest)