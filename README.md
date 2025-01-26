# Aplicação de Redes Neurais Recorrentes para Correspondência de Dados de Sísmica Time-Lapse


## Resumo

A sísmica 4D é uma técnica amplamente utilizada na indústria de óleo e gás para monitorar alterações em reservatórios, oferecendo informações sobre as mudanças nas propriedades de reservatórios, porém, a não repetibilidade entre diferentes experimentos sísmicos resulta em ruídos 4D, que tornam difícil a interpretação das mudanças geológicas.

Para reduzir esses ruídos, este estudo desenvolveu uma rede neural baseada em GRU que aprende a correspondência entre dados sísmicos simulados que seriam adquiridos em diferentes momentos (dados monitor) e dados iniciais de referência (dados baseline). A metodologia incluiu o treinamento da rede com diferentes tamanhos de batch e números de épocas, utilizando uma função de perda de erro quadrático médio.

Os resultados demonstraram que a rede neural proposta foi eficaz na redução dos ruídos nos dados sísmicos e na melhoria significativa da repetibilidade sísmica, medida pela métrica NRMS. Porém, por se tratar de uma inferência de padrões não lineares de longo prazo, a inferência  ainda não foi a esperada, sugerindo a necessidade de futuras melhorias no modelo, como o uso de camadas convolucionais 1D para aprimorar o aprendizado das dependências temporais. Os resultados alcançados mostram o potencial do uso de redes neurais para o processamento de dados sísmicos 4D para o monitoramento eficiente de reservatórios de petróleo em alto mar.


### Palavras-chave

Sísmica Time-Lapse, RNN, GRU, repetibilidade


# Title

Application of Recurrent Neural Networks for Data Matching in Time-Lapse Seismic


## Abstract

4D seismic is a widely used technique in the oil and gas industry to monitor changes in reservoirs, providing information about changes in reservoir properties. However, the non-repeatability between different seismic experiments results in 4D noise, making it difficult to interpret geological changes.

To reduce this noise, this study developed a neural network based on GRU that learns the correspondence between simulated seismic data that would be acquired at different times (monitor data) and initial reference data (baseline data). The methodology included training the network with different batch sizes and numbers of epochs, using a mean squared error loss function.

The results demonstrated that the proposed neural network was effective in reducing noise in the seismic data and significantly improving seismic repeatability, as measured by the NRMS metric. However, since it involves the inference of long-term nonlinear patterns, the inference was not yet as expected, suggesting the need for future model improvements, such as the use of 1D convolutional layers to enhance the learning of temporal dependencies. The results obtained show the potential of using neural networks for the processing of 4D seismic data for the efficient monitoring of offshore oil reservoirs.


### Keywords

Time-Lapse Seismic, RNN, GRU, Repeatability


## Introdução

Estimar as mudanças nas propriedades dinâmicas de reservatórios durante um período de produção a partir de dados de sísmica time-lapse (ou 4D) tem sido um desafio e uma ambição para geocientistas na indústria de óleo e gás [1]. Essas estimativas são atraentes para o monitoramento dos mesmos, e para o ajuste histórico, pois os dados sísmicos 4D oferecem informações sobre mudanças nas propriedades nos reservatórios inteiros, em um tempo específico de produção [1]. Mas o problema mora no fato de que esses elementos carregam, além de informações importantes, ruídos ocasionados pela não-repetibilidade entre diferentes experimentos sísmicos 3D, denominadas de ruídos 4D. O ruído 4D altera amplitude, fase e frequência das ondas sísmicas capturadas em diferentes experimentos ao longo do tempo. Essas alterações são uma superposição de todos os efeitos causados pelas variações simultâneas em qualquer propriedade dinâmica [1]. Esses ruídos podem ser desde mudanças na fonte sísmica até condições ambientais do local. Dessa forma, é importante projetar fluxos de aquisição e processamento sísmico para separar e atenuar os efeitos não repetíveis dos sinais repetíveis para identificar com precisão mudanças na estrutura geológica do subsolo [2].

Os dados utilizados neste trabalho são os chamados baseline, que são dados sísmicos coletados em um ponto inicial no tempo, antes de qualquer atividade de produção ou injeção no reservatório, servem como uma referência para comparar com dados adquiridos posteriormente, que por sua vez são chamados de dados monitor. Estes são capturados durante e após atividades de produção ou injeção no reservatório, sendo usados para observar as mudanças que ocorreram no reservatório ao longo do tempo.

O objetivo do processamento de dados sísmicos 4D é melhorar a repetibilidade dos dados sísmicos de diferentes épocas ajustando a amplitude sísmica, fase, assinatura da fonte, frequência e deslocamento temporal, e esse procedimento é chamado de equalização cruzada (XEQ, do inglês cross-equalization) [2], porém, fazendo uso de redes neurais recorrentes que é capaz de aprender a dependência dos dados.

Portanto, este trabalho teve como objetivo treinar uma rede neural que consiga produzir uma equivalência entre dados sísmicos adquiridos em diferentes instantes de tempo. Em particular, nosso interesse está em reduzir significativamente os ruídos presentes nesses dados, e a investigação é direcionada para o processo correspondência de dados 4D, buscando identificar e monitorar variações temporais na subsuperfície.


## Metodologia

Para o desenvolvimento deste trabalho foi necessário aprender como fazer uso de Recurrent Neural Networks (RNNs) que são necessárias para fazer a correspondência de dados de sísmica 4D, visto que é um problema em que os dados têm uma dependência temporal.

Uma Rede Neural Recorrente (RNN) é um tipo de rede neural artificial que utiliza estados internos, ou memória, para processar sequências de dados temporais. Porém, as RNNs tradicionais enfrentam o problema do desaparecimento do gradiente, o que dificulta o treinamento para longas sequências [7]. Para contornar essa limitação, foram desenvolvidas variantes de RNNs que introduzem mecanismos de portas que permitem controlar quais informações são mantidas ou descartadas ao longo do tempo, melhorando a capacidade da rede de aprender dependências de longo prazo, que são compostas por camadas totalmente conectadas e utilizam funções de ativação para gerenciar o fluxo de informações de maneira eficiente [7].

Para treinar uma RNN, o truque é desenrolá-la no tempo e, em seguida, usar a retropropagação regular, presente na figura 1. Essa estratégia é chamada de retropropagação no tempo (BPTT - Backpropagation Through Time) [5].




Figura 1 - Backpropagation ao longo do tempo.

Neste caso, os dados de entrada são os dados monitor, que são as medições obtidas em diferentes momentos, enquanto os dados de referência são os de saída, chamados de baseline. Os dados observados empiricamente, conhecidos como sismogramas, são registros feitos por geofones (em terra) ou hidrofones (no mar) das ondas acústicas geradas pelas fontes sísmicas e refletidas no interior da terra. 

O treinamento da rede é feito a partir de uma função que tem como entrada os traços, o tamanho de batch size e número de épocas escolhidas. A manipulação dos dados é feita a cada traço baseline e monitor, que são primeiramente normalizados com método Standard (padronização), transformando os valores de forma que tenham média zero e desvio padrão igual a um. Com a normalização é armazenada a escala dos dados baseline que vai ser útil para o teste, e após isso, é feito o janelamento de ambos. Para o presente trabalho, foi utilizado um tamanho de janela igual a 20, pois este valor é maior do que o período da wavelet que modela a fonte sísmica. Isso garante que as características principais da fonte sísmica sejam representadas dentro de cada janela, permitindo com que o modelo possa aprender melhor as relações temporais dos dados sísmicos.
Para o treinamento, foi definido que será utilizado 80% do dado, da amostra 0 até 800 (considerada parte rasa do dado), e os outros 20% para validação. A função de perda foi a do erro quadrático médio, na equação A a seguir.

(A)

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$


A parte profunda será predita no traço inteiro com o teste, que também contém algumas métricas de avaliação.
Uma maneira de obter uma estimativa do desempenho de generalização de um modelo é observar as curvas de aprendizado: são gráficos do desempenho do modelo no conjunto de treinamento e no conjunto de validação em função do tamanho do conjunto de treinamento (ou da iteração de treinamento) [5].
A repetibilidade sísmica pode ser medida através do NRMS (do inglês normalized rootmean-square), definido como a diferença de amplitude RMS dividido pela soma das médias  dos dois traços sísmicos em tempos diferentes [6]. 

A métrica RMS (do inglês Root Mean Square) é uma medida estatística comum usada para quantificar a magnitude de uma variação em um conjunto de dados. No contexto da sísmica, pode ser utilizada para medir a intensidade dos traços sísmicos e a magnitude das variações entre diferentes conjuntos de dados.

O NRMS, por sua vez, é uma forma normalizada do RMS que possibilita comparar a diferença relativa entre dois conjuntos de dados. A normalização é realizada ao dividir o RMS da diferença entre os dois conjuntos de dados pelo RMS da soma desses conjuntos e multiplicar o resultado por 200, o que converte o valor para uma porcentagem variando de zero a 200. O cálculo é feito a partir da equação B.

NRMS=200*RMS(B-M)RMS(B)+RMS(M)
(B)


Foi utilizada uma célula GRU (Gated Recurrent Unit) para prever o comportamento dos dados baseline a partir dos dados monitor. Para isso, os dados monitor serão usados como entrada, enquanto os dados baseline serão usados como saída. Ela foi proposta por Cho et al. [2014] para permitir que cada unidade recorrente capture de forma adaptativa dependências em diferentes escalas de tempo, possuindo unidades de porta que modulam o fluxo de informação dentro da unidade, porém, sem ter células de memória separadas [3].
A GRU utiliza dois portões, um para realizar a atualização e outro para a redefinição da informação, como demonstrado na figura 2 [4]. Resumidamente ela é composta por quatro etapas, dentre elas a (i) atualização do portão permitindo definir quanto das informações anteriores precisa ser passada para a próxima etapa, (ii) redefinição do portão definindo quanto das informações anteriores serão esquecidas pela rede, (iii) conteúdo atual da memória, que define qual o conteúdo da memória nesse instante, quando utilizará a saída da redefinição para guardar informações relevantes do passado e a (iv) saída da unidade oculta, etapa que tem a memória final, definindo o valor que saíra como resultado sobre etapas anteriores [4].

Figura 2 - Uma unidade da RNN-GRU [Geron 2019]

Embora as células LSTM e GRU possam lidar com sequências muito mais longas do que as RNNs simples, elas ainda têm uma memória de curto prazo relativamente limitada e têm dificuldade em aprender padrões de longo prazo em sequências de 100 ou mais etapas de tempo, como amostras de áudio, séries temporais longas ou frases longas [5].

Resultados e Discussão
1/20000
Apresentação dos dados coletados e confronto dos resultados com a literatura consultada.

Os testes foram realizados para (i) três traços (do traço 199 ao 201) e para (ii) cinco traços (do traço 198 ao 202), ambos com 300 épocas, e diferentes tamanhos de batch size.

O treinamento com três traços teve como entrada os três traços centrais [199, 200, 201] dos dados monitor, e um dos diferentes valores para o parâmetro batch size, o que resultou em uma menor diferença (ou seja, melhor resultado) dos traços inferidos em relação ao monitor perfeito foi o igual a 700, após o teste da rede foram montados gráficos do traço 199 nas Figuras 3a e 3b, do gráfico 200 nas Figuras 4a e 4b e do gráfico 201 nas Figuras 5a e 5b com o intuito de analisar as diferenças entre os traços monitor, baseline e inferido em relação ao monitor perfeito.




Figuras 3 - (a) Região do reservatório e (b) Traço 199 inteiro.




Inteiro
Chegada
Profundo
Reservatório
Baseline
0,43
0,0
9,27
9,74
Monitor
13,64
13,65
9,21
8,61
Inferido
0,34
0,31
3,07
2,6


Tabela 1 - Diferenças entre as diferentes regiões dos traços 199 em relação ao monitor perfeito.




Figuras 4 - (a) Região do reservatório e (b) Traço 200 inteiro.




Inteiro
Chegada
Profundo
Reservatório
Baseline
0,44
0,0
9,44
9,88
Monitor
14,02
14,03
9,27
8,66
Inferido
0,34
0,31
3,01
2,55


Tabela 2 - Diferenças entre as diferentes regiões dos traços 200 em relação ao monitor perfeito.



Figuras 5 - (a) Região do reservatório e (b) Traço 201 inteiro.



Inteiro
Chegada
Profundo
Reservatório
Baseline
0,44
0,0
9,61
10,03
Monitor
14,41
14,42
9,33
8,74
Inferido
0,64
0,63
2,95
2,5


Tabela 3 - Diferenças entre as diferentes regiões dos traços 201 em relação ao monitor perfeito.

A curva de treinamento dos três traços com o batch size igual a 700 na Figura 6 indica que tanto a perda de treinamento quanto a perda de validação estão diminuindo e permanecem próximas uma da outra ao longo das épocas, o modelo está aprendendo corretamente e não está sobreajustando.


Figura 6 - Curva de treinamento dos traços 199, 200 e 201.


Já o treinamento com cinco traços teve como entrada os três traços centrais [198, 199, 200, 201, 202] dos dados monitor, e valor igual a 700 para o parâmetro batch size, que resultou em uma menor diferença (portanto, melhor resultado) dos traços inferidos em relação ao monitor perfeito. Após o teste da rede, foram montados gráficos de todos os cinco traços, mas a fim de comparação serão mostrados apenas o traço 198 nas Figuras 7a e 7b, o traço central 200 nas Figuras 8a e 8b e o traço 201 nas Figuras 9a e 9b com o intuito de analisar as diferenças entre os traços monitor, baseline e inferido em relação ao monitor perfeito.


Figuras 7 - (a) Região do reservatório e (b) Traço 198 inteiro.



Inteiro
Chegada
Profundo
Reservatório
Baseline
0,42
0,0
9,1
9,61
Monitor
13,26
13,26
9,15
8,57
Inferido
0,71
0,7
2,79
2,19


Tabela 4 - Diferenças entre as diferentes regiões dos traços 198 em relação ao monitor perfeito.



Figuras 8 - (a) Região do reservatório e (b) Traço 200 inteiro.




Inteiro
Chegada
Profundo
Reservatório
Baseline
0,44
0,0
9,44
9,88
Monitor
14,02
14,03
9,27
8,66
Inferido
0,46
0,44
2,72
2,12


Tabela 5 - Diferenças entre as diferentes regiões dos traços 200 em relação ao monitor perfeito.


Figuras 9 - (a) Região do reservatório e (b) Traço 202 inteiro.





Inteiro
Chegada
Profundo
Reservatório
Baseline
0,45
0,0
9,77
10,19
Monitor
14,8
14,81
9,39
8,83
Inferido
1,06
1,05
2,64
2,05


Tabela 6 - Diferenças entre as diferentes regiões dos traços 202 em relação ao monitor perfeito.



A curva de treinamento dos três traços com o batch size igual a 2000 na Figura 10 indica que tanto a perda de treinamento quanto a perda de validação estão diminuindo e permanecem próximas uma da outra ao longo das épocas, o modelo está aprendendo corretamente e não está sobreajustando. Na última época, o comportamento das curvas indicam que poucas épocas à frente o modelo ainda iria aprender mais.


Figura 10 - Curva de treinamento dos traços 198, 199, 200, 201 e 202.




## Conclusões

Os resultados obtidos dos traços inferidos presentes nas tabelas em ambos os testes foram bons, visto que foram perto de zero, e todos os valores na região do reservatório dos traços inferidos - em relação ao monitor perfeito - foram menores do que os valores do baseline respectivo. Em todos os gráficos gerados da região do reservatório é possível perceber que a inferência do traço ficou bem aproximada do ideal enquanto os ruídos foram reduzidos. Já no gráfico dos traços inteiros fica perceptível que na região da onda de chegada as amplitudes foram consideravelmente reduzidas mas ainda não da maneira ideal que seria zero.

A curva de aprendizagem para o treino dos traços 199, 200 e 201 - na Figura 6 - ficou semelhante ao da Figura 10 que foi referente ao treino com cinco traços, ambos indicam que na última época, os modelos já não estavam aprendendo muito, e uma possibilidade para melhorar seria de testar novos valores de parâmetros. Utilizar o método Bayesian para testar diferentes valores de parâmetros pode ser uma solução futura.

Comparando o treino de três com o de cinco traços, a diferença encontrada entre os testes é que, ao aumentar a quantidade de traços no treinamento - e dessa forma, aumentando a quantidade de dados de entrada - a inferência dos traços “dos extremos” acabam resultando em erros maiores. No caso do treino com cinco traços, a inferência do 198 e 202 teve um erro um pouco maior. No geral, os resultados foram bons, mas como Geron diz em seu livro, uma possível maneira de resolver a dificuldade em aprender padrões de longo prazo em sequências de 100 ou mais etapas de tempo, é encurtar as sequências de entrada, por exemplo, usando camadas convolucionais 1D.

Agradecimento pela colaboração dos professores Dr. Gilberto Corso, Dr. Tiago Barros e Ramon Araújo, e em especial ao CNPq pelo apoio financeiro através do financiamento da bolsa de iniciação científica.
	


## Referências

[1] Côrte, G., Dramsch, J., Amini, H., and MacBeth, C. (2020). Deep neural network application for 4D seismic inversion to changes in pressure and saturation: Optimizing the use of synthetic training datasets. Geophysical Prospecting, 68(7):2164–2185.

[2] Jun, H. and Cho, Y. (2022). Repeatability enhancement of time-lapse seismic data via a convolutional autoencoder. Geophysical Journal International, 228(2):1150–1170. Number: 2.

[3] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. In NIPS 2014 Workshop on Deep Learning, December 2014
https://arxiv.org/pdf/1412.3555 

[4] Souza, Renato Manoel de Reconhecimento de emoções através da fala utilizando redes neurais / Renato Manoel de Souza ; orientador, Mauro Roisenberg, 2020. https://repositorio.ufsc.br/bitstream/handle/123456789/218146/TCC.pdf?sequence=1&isAllowed=y

[5] GERON, Aurélien. Mãos à obra: machine learning com Scikit-Learn, Keras & TensorFlow. 2. ed. São Paulo: Alta Books, 2020.

[6] PINTO, José Rodrigo Dias Oliveira. Estimativa do valor da informação da sísmica 4D. 2009.

[7] Alali, A., Kazei, V., Sun, B., Smith, R., Nivlet, P., Bakulin, A., & Alkalifah, T. (2020). Cross-equalization of time-lapse seismic data using recurrent neural networks. SEG Technical Program Expanded Abstracts 2020. doi:10.1190/segam2020-3424773.1