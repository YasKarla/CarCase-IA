## CarCase IA

CarCase IA é um projeto mobile baseado em Inteligência Artificial e Realidade Aumentada, que permite identificar automaticamente o modelo de um carro utilizando a câmera do dispositivo. Após o reconhecimento, o sistema gera uma ficha técnica completa do veículo, com dados como motor, potência, consumo, combustível e ano.

O modelo de IA foi treinado utilizando a arquitetura YOLOv8, com um dataset customizado de veículos. Os dados técnicos foram extraídos do arquivo cars_meta.mat, convertidos para CSV e utilizados para correlacionar as classes do modelo com informações técnicas detalhadas.

## Objetivo

O objetivo do projeto é oferecer uma experiência interativa e informativa para o usuário, permitindo que ele aponte a câmera para um carro e receba, em tempo real, a ficha técnica do modelo reconhecido. A solução é ideal para aplicações educacionais, exposições automotivas, concessionárias e sistemas de suporte a manutenção.

## Arquitetura da Solução

A solução foi construída em três camadas principais:

Modelo de IA (YOLOv8):

- Responsável por detectar e classificar o modelo do carro na imagem capturada pela câmera.
- Treinado com imagens rotuladas de diversos modelos automotivos.
- Exportado em formato .pt para uso em inferência remota.

API de Inferência:
- Desenvolvida em Python, utilizando FastAPI.
- Realiza a inferência do modelo YOLOv8 e retorna o nome do carro detectado.

A partir do resultado, busca automaticamente no CSV extraído do cars_meta.mat os dados técnicos correspondentes.
Resposta JSON contém:
{
  "modelo_detectado": "Lamborghini_Diablo_Coupe_2001",
  "ficha_tecnica": {
    "ano": 2001,
    "motor": "5.7 V12",
    "potencia": "530 hp",
    "consumo": "6 km/l",
    "combustivel": "Gasolina"
  }
}
