# vllm_model.py
from vllm import LLM, SamplingParams,LLMEngine
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer,AutoConfig
import os
import json
import pandas as pd
import ast
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE']='True'

# def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=10000):
#     stop_token_ids = [151329, 151336, 151338]
#     # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
#     sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
#     # 初始化 vLLM 推理引擎
#     llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True,enforce_eager=True,use_rope_scaling=True)
#     # engin = LLMEngine(model)
#     outputs = llm.generate(prompts, sampling_params)
#     return outputs
def clean_text(text):
    """
    清理文本中的换行符和多余空格
    """
    # 1. 将文本按行分割并清理每行
    lines = text.splitlines()  # 比 split('\n') 更好，可以处理不同操作系统的换行符

    # 2. 清理每行的空白字符并过滤空行
    cleaned_lines = []
    for line in lines:
        # 清理每行首尾的空白字符
        line = line.strip()
        # 将行内的多个空白字符替换为单个空格
        line = ' '.join(line.split())
        if line:  # 只保留非空行
            cleaned_lines.append(line)

    # 3. 将所有行连接成一个字符串，用空格分隔
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = f'"{cleaned_text}"'

    return cleaned_text

def get_completion(prompts, model, tokenizer=None, max_tokens=100, temperature=0.1, top_p=0.9,max_model_len=20000):
    stop_token_ids = [151329, 151336, 151338]

    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )

    # 创建引擎参数
    # config = AutoConfig.from_pretrained(model)

    # 初始化 LLM
    # llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True,enforce_eager=True,engine_args=True)
    # llm = LLM(
    #     model=model,
    #     tokenizer=model,
    #     trust_remote_code=True,
    #     max_seq_len_to_capture=max_model_len,
    #     gpu_memory_utilization=0.99,
    #     # model_config=config
    #     rope_scaling={
    #     "factor":4.0,
    #     "original_max_position_embeddings":32768,
    #     "type":"yarn"}
    # )
    llm = LLM(
        model=model,
        tokenizer=model,
        trust_remote_code=True,
        max_model_len=80000,  # 减小这个值
        gpu_memory_utilization=0.95,
        rope_scaling={
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
    )


    # 生成文本
    outputs = llm.generate(prompts, sampling_params)

    return outputs

if __name__ == "__main__":
    # 初始化 vLLM 推理引擎
    # model='./Qwen/Qwen2___5-14B-Instruct' # 指定模型路径
    # model="Qwen2___5-7B-Instruct" # 指定模型名称，自动下载模型
    # model = "/home/sidney/models_llm/hub/Qwen/Qwen2___5-14B-Instruct"
    model = "/home/sidney/models_llm/hub/Qwen/Qwen2___5-7B-Instruct"
    tokenizer = None

    # 加载分词器后传入vLLM 模型，但不是必要的。
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    template = """
    Select the most accurate publication date from the given date list and reference text:
    
    examples：
    
    Example 1:
    [Date List]
    ['23/11/2023','10 mars 2023','15 janvier 2024','31 janvier','15 janvier 2024','31 janvier 2024','en février 2024 et','le 01/12/2023','le 01/12/2023','le 01/12/2023']
    [Reference Text]
    "https://www.cc-bocage-bourbonnais.com/images/Délibération_ZA_Enr/20231130_Tronget.pdfCOMMUNE DE TRONGET EXTRAIT DU REGISTRE DES DÉLIBÉRATIONS République Française L’an deux mil vingt-trois, le jeudi 30 novembre 2023 à 19h30, le Conseil Municipal, Département de l’Allier dûment convoqué, s’est réuni salle de la Mairie sise 8 passage de la mairie, en session Arrondissement de Moulins ordinaire, sous la présidence du Maire, Jean-Marc DUMONT. Date de convocation : Présents : Patrick AMATHIEU, Elena BARANSKI, Daniel CANTE, Alain DETERNES, 23/11/2023 Jean-Marc DUMONT, Audrey GERAUD, Patricia RAYNAUD, Pascal RAYNAUD, Sylvain RIBIER, Franck VALETTE Nombre de conseillers : Excusés : Laurent BRUN, Jean-Marc CARTE, Stéphane HERAULT, Annie WEGRZYN En exercice : 14 Présents : 10 Votants : 14 Le quorum étant atteint, le | Pouvoirs: Laurent BRUN à Franck VALETTE, Jean-Marc CARTE à Jean-Marc Conseil Municipal peut | DUMONT, Stéphane HERAULT à Pascal RAYNAUD, Annie WEGRZYN à Eléna valablement délibérer. BARANSKI Secrétaire de séance : Daniel CANTE Zones d’Accélération des Energies Renouvelables N°36/2023 La loi n° 2023-175 du 10 mars 2023 relative à l’accélération de la production d’énergies renouvelables, dite loi APER, vise à accélérer et simplifier les projets d’implantation de producteurs d’énergie et à répondre à l’enjeu de l’acceptabilité locale. En particulier, son article 15 permet aux communes de définir, après concertation avec leurs administrés, des zones d’accélération où elles souhaitent prioritairement voir des projets d’énergies renouvelables s’implanter. Les zones d’accélération (ZAENR) concernent ainsi l’implantation d'installations terrestres de production d’énergies renouvelables, ainsi que de leurs ouvrages connexes. Ces ZAENR peuvent concerner toutes les énergies renouvelables (ENR). Elles sont définies, pour chaque catégorie de sources et de types d’installation de production d’ENR, en tenant compte de la nécessaire diversification des ENR, des potentiels du territoire concerné et de la puissance d’ENR déjà installée. (L.141-5-3 du code de l’énergie) Ces zones d’accélération ne sont pas des zones exclusives. Des projets pourront être autorisés en dehors. Toutefois, un comité de projet sera obligatoire pour ces projets, afin de garantir la bonne inclusion de la commune d’implantation et des communes limitrophes dans la conception du projet, au plus tôt et en continu. Les porteurs de projets seront, quoi qu’il en soit, incités à se diriger vers ces ZAENR qui témoignent d’une volonté politique et d’une adhésion locale du projet ENR. Monsieur le Maire précise que : ° Pour un projet, le fait d’être situé en zone d’accélération ne garantit pas son autorisation, celui-ci devant, dans tous les cas, respecter les dispositions réglementaires applicables et en tout état de cause l’instruction des projets reste faite au cas par cas. + Les zones doivent être à faibles enjeux environnementaux, agricoles et paysagers. + L’article L.314-41. du code de l’énergie prévoit que les candidats retenus à l’issue d’une procédure de mise en concurrence ou d’appel à projets sont tenus de financer notamment des projets portés par la commune ou par l’établissement public de coopération intercommunale à fiscalité propre d’implantation de l’installation en faveur de la transition énergétique. + Les communes identifient par délibération du conseil municipal des zones qui sont soumises à concertation du public selon les modalités qu’elles déterminent librement. Compte tenu de ces éléments, Monsieur le Maire expose : Les propositions de zones d’accélération pour les énergies renouvelables se fondent sur les critères suivants : + Des délaissés d’infrastructures, + __ Des zones dégradées, + Des terres agricoles inexploitables, + La présence de projets déjà connus, Les ZAENR proposées à la concertation sont les suivantes : + __ Solaire photovoltaïque : sur l’ensemble des bâtiments communaux et domaine public + __ Solaire photovoltaïque au sol dont ombrières : sur le domaine public et biens publics + _ Éolien, méthanisation : pas détermination de zone + Réseau de chaleur, bois-énergie, géothermie : projets communaux ou publics Les modalités de concertation proposées sont les suivantes : + Mise à disposition des documents et d’un registre en mairie du 15 janvier 2024 au 31 janvier 2024. + Mise à disposition des documents et d’un formulaire sur le site internet de la Communauté de Communes du Bocage Bourbonnais du 15 janvier 2024 au 31 janvier 2024. Le conseil municipal procédera à l’élaboration d’un bilan de la concertation en février 2024 et apportera les éventuelles modifications aux propositions des zones d’accélération des énergies renouvelables. Monsieur le Maire propose donc au conseil municipal d’émettre un avis favorable à : + La proposition de ZAENR pour leur mise en concertation du public, + La proposition des modalités de concertation. Après en avoir délibéré et à l'unanimité des membres présents et représentés, le Conseil Municipal décide : + D'’identifier les zones d’accélération pour l’implantation d’installations terrestres de production d’énergies renouvelables ainsi que leurs ouvrages connexes mentionnées ci- après, ainsi que sur les cartes annexées à la présente décision, qui seront soumises à concertation du public ; + Valide les modalités de concertation ; + Charge le maire ou son représentant de transmettre à l’EPCI, les zones identifiées pour concertation du public. ONT VOTE POUR : 14 ONT VOTE CONTRE : / SE SONT ABSTENUS : / ACTE EXECUTOIRE Reçu par le représentant de l’Etat le 01/12/2023 et publié le 01/12/2023 Pour extrait conforme au registre des délibérations du conseil municipal, Fait à Tronget, le 01/12/2023 Le Maire, UNS Jean-Marc DUMONT"
    Answer:31 janvier 2024
    
    Example 2: 
    [Date List]
    ['le 06/01/2024','05 Janvier 2024','03-2024','10 mars 2023','10 mars 2023','le 06/01/2024','le 06/01/2024','19/12/2023','30/12/2023','le 05 Janvier 2024']
    [Reference Text]
    "https://mairiechars95.fr/wp-content/uploads/2024/03/Deliberation-3-Zone-dacceleration-des-energies-renouvelables.pdfEnvoyé en préfecture le 06/01/2024 RÉPUBLIQUE FRANÇAISE M À | FR | E D FC Reu en préfecturé1e06/0172024 VAL-D'OISE | Publié le ID : 095-219501426-20240105-032024-DE Extrait du registre des délibérations du conseil municipal de CHARS Séance du 05 Janvier 2024 03-2024 OBJET : Décision du conseil municipal sur les zones d'accélération des énergies renouvelables Présents : 15 Evelyne BOSSU Xavier BACHELET Ariane MARTIN Carole BOUILLONNEC Vincent DELCHOQUE Jean-Pierre BAZIN Sébastien RAVOISIER Sheila DEPUILLE Pierre-Antoine DHUICQ Patricia CHAILLOU-LEPAREUR Sylviane LEPAPE Gérard GENNISSON Nathalie GROM Philippe CHAUVET Nicolas BELANGÉ Absents et procurations : 4 Sandrine LHORSET excusée Nicolas PRIOUX excusé Agnès AGLAVE-LUCAS excusée Caroline BOURG Pouvoir à Sylviane LEPAPE Le Président a ouvert la séance et fait l’appel nominal, il a été procédé en conformité avec l’article L.2121-15 du code général des collectivités territoriales, à la nomination d'un secrétaire pris au sein du conseil. Monsieur Philippe CHAUVET est désigné pour remplir cette fonction. Pour rappel : La loi n° 2023-175 du 10 mars 2023 relative à l'accélération de la production d'énergies renouvelables vise à accélérer le développement des énergies renouvelables de manière à lutter contre le changement climatique et préserver la sécurité d'approvisionnement de la France en électricité. L'article 15 de la loi a introduit dans le code de l'énergie un dispositif de planification territoriale à la main des communes. D'ici la fin de l’année 2023, les communes sont invitées à identifier les zones d'accélération pour l'implantation d'installations terrestres de production d'énergie renouvelable. En application de l’article L141-5-3 du code de l'énergie, ces zones sont définies, pour chaque catégorie de sources et de types d'installation de production d'énergies renouvelables : éolien terrestre, photovoltaïque, méthanisation, hydroélectricité, géothermie, en tenant compte de la nécessaire diversification des énergies renouvelables en fonction des potentiels du territoire concerné et de la puissance des projets d'énergies renouvelables déjà installée. La zone d'accélération illustre la volonté de la commune d'orienter préférentiellement les projets vers des espaces qu'elle estime adaptés. Ces projets pourront bénéficier de mécanismes financiers incitatifs. En revanche, pour un projet, le fait d’être situé en zone d'accélération ne garantit pas la délivrance de son autorisation ou de son permis. Le projet doit dans tous les cas respecter les dispositions réglementaires applicables. Un projet peut également s'implanter en dehors des zones d'accélération. Dans ce cas, un comité de projet sera obligatoire. Ce comité inclura les différentes parties prenantes concernées par un projet d'énergie renouvelable, dont les communes limitrophes. Dans le cas où les zones d'accélération au niveau régional sont suffisantes pour atteindre les objectifs régionaux de développement des énergies renouvelables, la commune peut définir des zones d'exclusion de ces projets. Le conseil municipal de la commune de Chars, régulièrement convoqué, s'est réuni sous la présidence de Madame BOSSU Evelyne, afin de délibérer sur les zones d'accélération proposée par la commune sur son territoire. Madame le Maire constate que le conseil réunit les conditions pour délibérer valablement. Vu la loi n° 2023-175 du 10 mars 2023 relative à l'accélération de la production d'énergies renouvelables, notamment son article 15, 2, rue de Gisors 95750 Chars - Téléphone 01 30 39 72 36 - Télécopie 01 30 39 94 64 Envoyé en préfecture le 06/01/2024 Reçu en préfecture le 06/01/2024 Publié le ID : 095-219501426-20240105-032024-DE Madame le Maire présente les zones identifiées comme zones d'accélération pour le développement des énergies renouvelables ainsi que les arguments ayant conduit à ces propositions de zones. Conformément à la loi, une consultation du public a été effectuée du 19/12/2023 au 30/12/2023 selon les modalités suivantes : sur le site internet de la Commune et dossier consultable en libre d'accès en Mairie. - La Commune de Chars souhaite donc s'orienter principalement vers le développement de l’énergies solaire et a identifié, dans ce cadre, deux solutions : a/ Les ombrières photovoltaïques sur le parking de la gare, b/ Le photovoltaïque de toiture sur différents bâtiments communaux suffisamment dimensionnées pour accueillir des structures viables économiquement et sur l'ensemble des toitures de la ZA des 9 arpents. (détails des zones en annexe) Madame le Maire soumet cette proposition de zones à délibération. Suite à l'exposé de Madame le Maire et après avoir délibéré à l'unanimité des présents, le conseil municipal : - DEFINIT comme zones d'accélération des énergies renouvelables de la commune les zones proposées figurant en annexe à la présente délibération - __ VALIDE la transmission de la cartographie de ces zones à Madame le sous-préfet, référent préfectoral à l'instruction des projets d'énergies renouvelables et des projets industriels nécessaires à la transition énergétique, du département de Chars, ainsi qu’à la Communauté de Commune Vexin Centre dont elles sont membres. Certifié exécutoire À CHARS, le 05 Janvier 2024 compte tenu de la transmission Evelyne BOSSU, en sous-préfecture, le ….. et de la publication, le …"
    Answer: le 06/01/2024
    
    Current Task:
    [Date List]
    {事件列表}
    [Reference Text]
    {文本}
    
    Rules:
    1. Output exactly one date
    2. Output the same thing as in the Date List
    3. Return only the date without any explanation
    """

    dataframe_path = "./dataset_valid_ner.csv"
    dataframe = pd.read_csv(dataframe_path)
    file_list = dataframe['local_filename'].to_list()
    # print(file_list[:40])
    results_all ={}
    counter = 1
    total_file = len(file_list)
    for file in file_list:
        total_file -= counter
        print(f"now start processing {file}, left{total_file} files  ")
        row = dataframe[dataframe['local_filename']==file]
        context = row['text_content'].iloc[0]# 防止返回series
        # clean " \n"
        context_cleaned= clean_text(context)
        context_cleaned = context_cleaned[:4000]

        # context_string = ''.join(context)
        # print(context_cleaned)
        # break
        # context_ast = ast.literal_eval(context)
        # context_string = "".join(context_ast)

        # break
        timelist = row['extracted_dates'].iloc[0]
        # print(context)
        # time_new = ast.literal_eval(timelist)

        # timelist = time_new[:15]
        # print(type(timelist))
        # print(timelist)

        input = template.format(事件列表=timelist,文本=context_cleaned)
        print(input)
        # break
        outputs = get_completion(input, model, tokenizer=tokenizer, max_tokens=25, temperature=0.1, top_p=0.9)
        # # # print(outputs.output)
        for output in outputs:
            results = output.outputs[0].text
            print(results)
        results_all[file] = results
        # print(results_all)
        break
    # dataframe["predicted_time"] = dataframe['local_filename'].map(results_all)
    # dataframe.to_csv("final_results_predicted_14B_1112.csv",index=False)





    # text [prompt1,prompt2...]
    # messages = [
    #     {"role": "system", "content": "你是一个有用的助手。"},
    #     {"role": "user", "content": prompt}
    # ]
    # 作为聊天模板的消息，不是必要的。
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    # outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=20, temperature=1, top_p=1, max_model_len=2048)

    # # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # # 打印输出。
    # # for output in outputs:
    # #     prompt = output.prompt
    # #     generated_text = output.outputs[0].text
    # #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    # for output in outputs:
    #     print("="*50)
    #     print(f"Output Text: {output.outputs[0].text}")
    #     print(f"Generation Length: {len(output.outputs[0].text)}")
    #     print(f"Stop Reason: {output.outputs[0].finish_reason}")
    #     print("="*50)

#  text = [""""
# 请从以下时间列表中选择一个参考文本发布日期的时间:
# [时间列表]
# ['6 février 2023,', '25 JANVIER 2023','18 février 2023','6 février 2023 en','16 janvier 2023', '25 mars 2021','12 septembre 2022','14 novembre 2022','14/11/2022','31 août 2022','en 2023','2023, un','le 6 mars 2023']；

# [要求]
# 1. 只能从时间列表中选择一个时间作为答案
# 2. 不需要任何解释,只输出一个时间



# [参考文本]
# CONSEIL COMMUNAUTAIRE DU
# 25 JANVIER 2023
# PROCES VERBAL
# L’an deux mille vingt-trois, le vingt-cinq janvier à 18h30 heures, le Conseil Communautaire,
# légalement convoqué, s’est réuni au siège de la Communauté de Communes, salle Choisilles, sous la
# présidence de Monsieur le Président, Antoine TRYSTRAM.
# Présents :
# Beaumont-Louestault : M. Robert Jean-Paul ; Mme Frapier Sylvie ; M. Desjonquères Vincent
# Bueil-en-Touraine :
# Cerelles : M. Poulle Guy
# Charentilly : Mme Bouin Valérie
# Chemillé-Sur-Dême : M. Canon Eloi
# Epeigné-Sur-Dême : M. Goué Stéphane
# Marray : M. Capon Philippe
# Neuillé-Pont-Pierre : M. Jollivet Michel ; Mme Six Sylvie ; M. Savard Didier
# Neuvy-Le-Roi : M. Thélisson Flavien
# Pernay : M. Peninon Jean-Pierre ; Mme Barthélémy Karine
# Rouziers-de-Touraine : M. Behaegel Philippe ; Mme Dreux Danielle
# St-Antoine-du-Rocher : M. Grousset Francis ; M. Cornuault Patrick
# St-Aubin-le-Dépeint : M. Roger Sylvain
# St-Christophe-Sur-Le-Nais : Mme Lemaire Catherine ; M. Albert De Rycke Thierry
# St-Paterne-Racan : M. Lapleau Eric
# St-Roch : M. Anceau Alain ; Mme Jeudi Nicole
# Semblançay : M. Trystram Antoine ; Mme Hendrick Elsa ; Mme Plou Peggy
# Sonzay : M. Verneau Jean-Pierre
# Villebourg : M. Fromont Christophe
# Date de convocation : 18 janvier 2023
# Secrétaire de séance : Commune d’Epeigné-sur-Dème - M. Goué Stéphane
# Pouvoirs : Mme Pain Claude à M. Grousset Francis, Mme Soulier Karine à M. Lapleau Eric, Mme
# Groux Gisèle à M. Poulle Guy, Mme Goumont Isabelle à M. Verneau Jean-Pierre, M. Guyon Ghislain
# à Mme Bouin Valérie
# Excusés : M. Durand Benoît, M. Cornuault Patrick, M. Descloux Didier
# Séance enregistrée et retransmise via Facebook
# M. le Président présente ses vœux pour cette nouvelle année à l’ensemble des élus.
# Il précise que cette séance est importante car elle est celle de nos orientations budgétaires : il s’agit
# d’une première écriture de ce que sera notre année. M. le Président propose que l’on présente
# l’ensemble les points inscrits à l’ordre du jour et que l’on termine notre séance par le débat des
# orientations budgétaires.
# 1

# 1 - Adoption du procès-verbal du 9 Novembre et du 7 décembre 2022
# Ces deux procès-verbaux sont adoptés à l’unanimité.

# 2 – FINANCES
# A – Demandes de subventions – CRTE
# EAJE BEAUMONT LOUESTAULT
# CC02_2023 FINANCES CRTE - DEMANDE DE SUBVENTION EAJE BEAUMONT
# LOUESTAULT
# Monsieur le Président présente les éléments suivants :
# Dans le cadre de l’actualisation du dossier de demande de subvention pour l’Etablissement d’Accueil
# du Jeune Enfant sur la commune de Beaumont-Louestault, une mise à jour de la délibération présentée
# en séance du 8 décembre 2021, sous les références C197.2021, est nécessaire. En effet, il convient
# d’affiner le plan de financement et la répartition des taux sollicités par financeur.

# DETR – DSIL – DSID – FNADT – année 2023
# Collectivité

# CC Gâtine-Racan

# Opération

# C.T.I.6. EAJE Beaumont-Louestault
# Coût estimatif de l'opération
# Poste de dépenses

# (Les montants indiqués dans chaque poste de dépense
# doivent être justifiés)

# Montant prévisionnel HT
# 30 000.00 €

# Etudes et prestation d'assistance
# Maître d'ouvrage

# 80 000.00 €

# Travaux

# 890 000.00 €
# 0.00 €
# 0.00 €
# 0.00 €
# 0.00 €
# 0.00 €
# 0.00 €

# 1 000 000.00 €

# Coût HT

# Plan de financement prévisionnel
# Le cas échéant, joindre une copie des décisions d'octroi des subvention ou
# à défaut le courrier de dem ande

# Financeurs

# Sollicité ou acquis

# montant
# subventionnable
# H.T

# Taux
# intervention

# montant aide
# sollicité

# DETR

# sollicité

# 1 000 000.00 €

# 14.00%

# Conseil départemental 37

# sollicité

# 1 000 000.00 €

# 8.00%

# 80 000.00 €

# CRST

# sollicité

# 1 000 000.00 €

# 20.00%

# 200 000.00 €

# CAF

# acquis en 2021

# 1 000 000.00 €

# 38.00%

# 380 000.00 €

# 0.00%

# 800 000.00 €

# 200 000 €

# 20.00%

# 0.00 €

# 1 000 000.00 €

# 80.00%

# 800 000.00 €

# Sous-total des aides sollicitées
# Autofinancement (au – 20 % du coût du projet)

# Coût HT

# 140 000.00 €

# 2

# Le Conseil Communautaire, au regard de la présentation de Monsieur le Président, décide à
# l’unanimité de :
# -

# Valider l’opération, ses modalités et nouveau plan de financement tels que présentés cidessus,
# De donner tous pouvoirs à Monsieur le Président ou son représentant pour permettre
# l’application de la présente délibération et signature des documents inhérents à cette
# décision.

# TERRRAIN DE FOOTBALL SYNTHETIQUE
# CC01_2023 FINANCES CRTE - DEMANDE DE SUBVENTION TERRAIN DE FOOTBALL
# SYNTHETIQUE
# Monsieur le Président précise que dans le cadre de l’actualisation du projet « terrain de football
# synthétique » pour lequel les aménagements ont été revus à la baisse au regard des coûts trop
# importants en terme d’investissement, il est nécessaire de mettre à jour la délibération prise en séance
# du 8 décembre 2021 « équipements sportifs communautaires » Réf. C177.2021 pour y produire le
# nouveau plan de financement. (Budget modifié et nouvelle répartition des sollicitations par financeur).
# Cette mise à jour est demandée par certains organismes financeurs.

# 3

# DETR – DSIL – DSID – FNADT – année 2023
# Collectivité

# CC Gâtine-Racan

# Opération

# CT.I.2. . Equipement sportif intercommunal / Terrain
# Synthétique
# Coût estimatif de l'opération

# Poste de dépenses
# (Les montants indiqués dans chaque poste de dépense

# Montant prévisionnel HT

# doivent être justifiés)
# Terrain de football en gazon synthétique

# 795 500.00 €

# Eclairage

# 134 500.00 €

# Etudes

# 70 000.00 €
# 0.00 €
# 0.00 €
# 0.00 €
# 0.00 €
# 0.00 €
# 0.00 €

# 1 000 000.00 €

# Coût HT

# Plan de financement prévisionnel
# Le cas échéant, joindre une copie des décisions d'octroi des subvention ou
# à défaut le courrier de dem ande

# Financeurs

# Sollicité ou acquis

# montant
# subventionnable
# H.T

# Taux
# intervention

# montant aide
# sollicité

# DETR/DSIL

# sollicité en 2023

# 1 000 000.00 €

# 30.00%

# 300 000.00 €

# CD37

# sollicité en 2023

# 1 000 000.00 €

# 20.00%

# 200 000.00 €

# CRST

# sollicité en 2023

# 1 000 000.00 €

# 10.00%

# 100 000.00 €

# ANS

# sollicité en 2023

# 1 000 000.00 €

# 15.00%

# 150 000.00 €

# Fédération Française de Football

# sollicité en 2023

# 1 000 000.00 €

# 5.00%

# 50 000.00 €

# 0.00 €

# 0.00%

# 0.00 €

# 0.00 €

# 0.00%

# 0.00 €

# 0.00%

# 800 000.00 €

# 200 000 €

# 20.00%

# 0.00 €

# 1 000 000.00 €

# 80.00%

# 800 000.00 €

# Sous-total des aides sollicitées
# Autofinancement (au – 20 % du coût du projet)

# Coût HT

# Le Conseil Communautaire, au regard de la présentation de Monsieur le Président, à l’unanimité
# décide d’ :
# -

# Autoriser Monsieur le Président à déposer une demande de subvention pour l’opération cidessus exposée avec un plan ce financement actualisé et,
# Donner tous pouvoirs à Monsieur le Président ou son représentant pour permettre
# l’application de la présente délibération et signature des documents inhérents à cette
# décision.

# Monsieur le Président précise que trois délibérations supplémentaires sont présentées à l’approbation
# de l’assemblée délibérante :

# DEMANDE DE SUBVENTION - AIRES D’ACCUEIL DES GENS DU VOYAGE
# CC15_2023 FINANCES DEPOT DE DEMANDE DE SUBVENTION AIRES D’ACCUEIL DES
# GENS DU VOYAGE
# Monsieur le Président expose les éléments suivants :
# 4

# Deux aires de passage pour les gens du voyage ont été envisagées : une à Neuvy-le-Roi, une autre sur
# la commune de Semblançay.
# Monsieur le Président rappelle qu’une aire de passage est un équipement destiné à accueillir pour une
# durée temporaire, les gens du voyage, c’est-à-dire des personnes dont l’habitat traditionnel est la
# résidence mobile. L’objectif est d’accélérer la création de ces aires et ainsi améliorer les conditions de
# vie des gens du voyage sur le territoire.
# Il s’agit aussi d’éviter leur installation sur les infrastructures publiques et permettre une prise en charge
# des besoins des différentes communautés de manière plus limpide. La volonté est le développement de
# projets de réalisation d’aires avec des places assez larges et des blocs sanitaires en nombre suffisant
# tout en prenant en compte l’enjeu environnemental (maîtrise de l’énergie : réflexion sur l’installation de
# panneaux solaires, de récupération des eaux de pluie, isolation, structure bois, ...).
# Entendu la présentation de Monsieur le Président,
# Après en avoir délibéré, le Conseil Communautaire, à l’unanimité décide :
# -

# De valider le plan prévisionnel de financement au titre du CRST, inhérent aux aires de passage
# des gens du voyage tel que précisé comme ci-contre,

# -

# De donner tous pouvoirs à Monsieur le Président pour signer tous les documents inhérents à ce
# dossier.

# ZA LA BORDE – BEAUMONT LOUESTAULT - PROPOSITION
# D’EXTENSION – DEMANDE DE SUBVENTION
# CC14_2023 ACTION ECONOMIQUE - ZA LA BORDE – BEAUMONT LOUESTAULT PROPOSITION D’EXTENSION – DEMANDE DE SUBVENTION
# Monsieur le Président expose aux membres de l’assemblée délibérante les éléments suivants :
# La Communauté de Communes de Gâtine – Racan est gestionnaire de la zone d’activités La Borde –
# Beaumont-la-Ronce à Beaumont-Louestault.
# 5

# En 2013, une première extension du site, réalisée sur une emprise d’environ 1 ha, a permis de viabiliser
# 5 terrains à bâtir de 1 006 à 2 059 m2.
# La Communauté de Communes est propriétaire en continuité de la première extension d’une surface
# d’environ 6,22 hectares, destinée à l’extension du site d’activités.
# Il est proposé de réaliser des travaux d’aménagement d’ensemble de l’extension.
# Le plan de financement prévisionnel pour ces travaux s’établit comme annexé ci-contre.

# Le Conseil Communautaire, à l’unanimité, décide :
# -

# De valider le plan de financement prévisionnel repris ci-dessus,
# D’autoriser, Monsieur le Président ou son représentant, à solliciter une subvention au titre du CRST,
# D’autoriser, Monsieur le Président ou son représentant, à signer l’ensemble des documents
# nécessaires dans la mise en œuvre des actions liées à cette opération.

# ZA BEAU CLOS A PERNAY – TRAVAUX D’EXTENSION – DEMANDE DE
# SUBVENTION
# CC13_2023 ACTION ECONOMIQUE ZA BEAU CLOS A PERNAY – TRAVAUX
# D’EXTENSION - DEPOT DE DEMANDE DE SUBVENTION
# Monsieur le Président expose aux membres de l’assemblée délibérante les éléments suivants :
# La Communauté de Communes de Gâtine – Racan est gestionnaire de la zone d’activités Beau Clos à
# Pernay.
# La Communauté de Communes a vendu en août 2021 le dernier terrain viabilisé de cette zone.
# Des entreprises sont intéressées pour acquérir des parcelles sur cette zone d’activité. Pour permettre
# cette commercialisation, il est nécessaire de déposer un permis d’aménager et de prévoir des travaux
# notamment la réalisation d’une voirie permettant de desservir et viabiliser les futurs terrains.
# Il est donc proposé d’étudier l’extension du site d’activités.
# Le plan de financement prévisionnel pour la réalisation de l’extension de la ZA Beau Clos à Pernay
# s’établit comme annexé ci-contre
# Considérant la présentation de Monsieur le Président,
# 6

# Considérant la présentation du tableau portant le plan prévisionnel de l’opération,
# Le Conseil Communautaire, à l’unanimité, décide :
# -

# De valider le plan de financement prévisionnel repris ci-contre,
# D’autoriser Monsieur le Président ou son représenter à solliciter une subvention au titre du CRST
# cette opération,
# D’autoriser, Monsieur le Président ou son représentant, à signer l’ensemble des documents
# nécessaires dans la mise en œuvre des actions liées à cette opération.

# 3 – ACTION ECONOMIQUE
# A – Aide à l’investissement immobilier des entreprises
# Modification du cadre d’intervention de la communauté de communes
# CC04_2023 ACTION ECONOMIQUE - Aide à l’investissement des entreprises – Modification
# du cadre d’intervention de la Communauté de Communes
# Monsieur le Président laisse la parole à Monsieur CANON qui apporte les éléments suivants :
# Dans le cadre de l’aide à l’investissement immobilier et donc du CAP Immobilier, le Conseil Régional
# a revu ses pratiques pour essayer d’accompagner au mieux le projet de l’entreprise.
# La Région accompagne, désormais, les projets dont le capital de la SCI est détenu majoritairement par
# l’entreprise qui exploite le lieu (entreprise individuelle ou forme sociétaire). La Région n’intervient
# plus pour les dossiers dont le capital social est détenu majoritairement par une personne physique.
# La Région a souhaité redéfinir la façon d’accompagner les SCI, notamment les SCI détenues
# majoritairement par des personnes physiques car ce type de soutien aux SCI détenues majoritairement
# par des personnes physiques favorisaient la constitution d’un patrimoine personnel des détenteurs des
# actions de la SCI, ce qui n’est pas l’objet de l’aide publique.
# Compte-tenu de ce nouveau positionnement de la Région, il convient de redéfinir le positionnement de
# la Communauté de Communes.
# 7

# La Commission Economie, réunie le 4 janvier 2023, propose de continuer à accompagner les projets
# des entreprises au titre de l’aide à l’investissement à l’immobilier qui seront soutenus et accompagnés
# par la Région Centre Val de Loire, c’est-à-dire les projets des entreprises dont le capital social sera
# détenu majoritairement par l’entreprise qui exploite le lieu (entreprise individuelle ou forme sociétaire).
# Vu l’avis de la Commission Economie,
# Le Conseil Communautaire à l’unanimité décide :
# - de valider le principe que la Communauté de Communes Gâtine – Racan accompagne uniquement,
# au titre de l’aide à l’investissement immobilier, les projets des entreprises qui seront soutenus par la
# Région Centre Val de Loire soit les projets des entreprises dont le capital social sera détenu
# majoritairement par l’entreprise qui exploite le lieu (entreprise individuelle ou forme sociétaire),
# - donner pouvoir à Monsieur le Président ou son représentant pour signer tout document permettant
# la mise en application de la présente délibération

# B - POLAXIS – Demande RUSTIN – Lot 24
# CC05_2023 ACTION ECONOMIQUE - POLAXIS - Demande de l’entreprise RUSTIN- Lot 24
# Monsieur le Président laisse la parole à M. CANON qui expose les éléments suivants :
# La SARL ETABLISSEMENTS L. RUSTIN a acquis, via la SCI ELLO, en 2020, auprès de la
# Communauté de Communes Gâtine – Racan, le lot n°23, parcelles cadastrées ZK n°42 et ZK n°45, du
# parc d’activités POLAXIS représentant une superficie de 22 331 m2, pour y développer une nouvelle
# unité de production dédiée au silicone.
# Un bâtiment de 1 500 m2 extensible y a été construit en 2020, comprenant un atelier de production et
# des bureaux/locaux sociaux.
# L’entreprise vient de déposer un nouveau permis de construire pour étende l’unité de production pour
# une superficie complémentaire d’environ 1 400 m2.
# Pour répondre à un fort développement de l’entreprise, Monsieur RUSTIN, a formulé le souhait
# d’acquérir, le lot n°24, soit les parcelles cadastrées ZK n°74 et ZK n°76, d’une superficie totale de
# 7 311 m2, jouxtant le terrain qu’occupe déjà son entreprise. L’objectif est d’y construire un bâtiment
# qui permettra la réalisation des mélanges de l’entreprise.
# Suite à délibération communautaire, le prix du lot n°24 est de 30,00 € H.T le m2. Il était de 21,00 € H.T
# le m2 avant juin 2022.
# Pour rappel, la SCI ELLO a acquis le lot n° 23 de 22 331 m2 le 11 mars 2020 au prix de 18 € H.T le
# m2.
# Monsieur le Président rappelle que le sujet a été abordé en bureau communautaire du 12 janvier 2023 et
# qu’après échanges entre élus, ce dernier est autorisé à mener une négociation à hauteur de 25 euros le
# m2,
# L’idée est de se caler sur le tarif des terrains les mieux placés
# Vu la demande de la SARL ETS L. RUSTIN en date du 14 décembre 2022,
# Le Conseil Communautaire accepte à la majorité avec une abstention (Mme PLOU) d’autoriser :
# -

# La vente du lot n°24 soit les parcelles cadastrées ZK n°74 et ZK n°76 d’une superficie totale de 7 311
# m2 à la SCI ELLO, gérée par Mr et Mme RUSTIN, au prix de 25 euros le m2 ;

# -

# Monsieur le Président ou son représentant, à signer tous les documents afférents à ce dossier.

# 8

# C - Le Prisme Coworking – Règlement intérieur
# CC06_2023 ACTION ECONOMIQUE - APPROBATION REGLEMENT INTERIEUR COWORKING « LE PRISME »
# Il est proposé par délibération, l’adoption d’un règlement intérieur suite à la réception du coworking.
# Le bâtiment est réceptionné. Morgane est arrivée et est en charge du lieu de vie accueil des
# entreprises…elle travaille avec le service Communication.
# Il faut adopter le règlement pour qu’il devienne applicable. C’est un premier jet et il peut évoluer au fil
# des semaines.
# Les premiers clients se sont présentés ce matin ; il s’agissait d’une réunion au sein de notre bâtiment, de
# la caisse locale agricole.
# Le règlement s’applique aux co workeurs.
# Nous restons dans l’attente des logiciels pour les paiements etc….
# M. le Président indique qu’il est nécessaire de valider le projet de règlement intérieur inhérent au
# fonctionnement du co-working, cela afin de permettre son application auprès des utilisateurs qui
# souhaitent bénéficier de la structure.
# Ce document permet de définir les modalités d’utilisation de cet espace de travail partagé. Il précise
# également les tarifs liés à sa location.
# M. le Président précise que si le document nécessite une quelconque modification ultérieure, le
# règlement sera à nouveau présenté devant l’assemblée du Conseil Communautaire pour y être
# entérinée.
# Entendu la présentation de Mr le Président,
# Le Conseil Communautaire décide à l’unanimité de :
# -

# Valider la version du règlement intérieur du « Prisme » ainsi présentée en séance,

# -

# d’autoriser, Monsieur le Président ou son représentant, à signer tous les documents afférents à ce
# dossier.

# D - Parc photovoltaïque sud POLAXIS SAS EneR37 – Signature contrat inter
# créanciers
# CC07_2023 ACTION ECONOMIQUE - Parc photovoltaïque sud POLAXIS SAS EnerR 37 Signature contrat inter créanciers
# Monsieur le Président informe les membres de l’assemblée délibérante des éléments suivants :
# Le conseil communautaire du 29 juin 2022 a validé l’entrée de la Communauté de Communes Gâtines-Racan au
# capital de la SAS ENER37 à hauteur de 10%. Cette société de projets a été créée en partenariat avec le SIEIL et
# EneR CENTRE-VAL DE LOIRE pour le développement, la construction et l’exploitation des centrales de
# Neuillé-Pont Pierre Sud et Nord.
# Dans le cadre du financement de la construction de la centrale Neuillé-Pont Pierre Sud, la SAS ENER37
# souhaite en qualité d’emprunteur (ci-après l’« Emprunteur ») conclure avec la Banque Populaire (le
# « Prêteur »), une convention de crédit (ci-après la « Convention de Crédit ») dont les principales
# caractéristiques sont les suivantes :
# - Montant du crédit long terme : 3.300.000 € (+/- 100.000 €)
# - Montant du crédit TVA :
# 800.000 €
# - Taux d’intérêt :
# 3,80 %
# - Durée :
# 20 ans
# - Gearing :
# 85 % / 15 %

# 9

# -

# Soit des apports en CCA :

# 600.000 €

# Les garanties exigées dans le cadre de la Convention de Crédit :
# − Nantissement des compte-titres en 1er rang portant sur l’intégralité des titres de l’Emprunteur
# − Nantissement des comptes bancaires de la SAS ENER37
# − Cession Dailly des créances au titre :
# * du contrat d’achat de l’électricité
# * des indemnités dues et à devoir dans le cadre des contrats de projets (construction, maintenance)
# * des polices d’assurance en phase de construction et en phase d’exploitation souscrites par
# l’Emprunteur
# * des créances TVA
# − Gage sans dépossession des matériels
# (Ci-après ensemble les « Sûretés »).
# Les Sûretés et la Convention de Crédit sont ci-après dénommés ensemble les « Documents de Financement ».
# (1) En garantie des obligations de l’Emprunteur au titre de la Convention de Crédit, il est envisagé que la
# Communauté de Communes Gâtines-Racan en qualité d’actionnaire s’engage à nantir en premier rang les actions
# qu’il détient dans le capital social de l’Emprunteur en faveur du Prêteur (le « Contrat de Nantissement du
# Compte-Titres Financiers ») (en ce inclus la déclaration de nantissement y afférente).
# Il est également prévu qu’un accord Intercréanciers (ci-après l’ « Accord Intercréanciers ») entre le Prêteur,
# l’Emprunteur et les actionnaires de la SAS ENER37 soit conclu afin d’encadrer notamment :
# -

# La subordination des paiements et créances de la SAS ENER37 au Prêteur,
# L’engagement des actionnaires de la SAS ENER37 d’apports en fonds propres et apports en fonds
# propres complémentaires sous réserve du respect des dispositions du code général des collectivités
# territoriales,
# L’engagement des actionnaires de la SAS ENER37 de maintenir leur participation au capital de la SAS
# ENER37.

# Les administrateurs sont appelés à statuer ce jour sur :
# -

# L’examen des termes et conditions et autorisation de la conclusion par le SIEIL en qualité d’actionnaire
# de la SAS ENER37 (i) du Contrat de Nantissement du Compte-Titres Financiers et (ii) de l’Accord
# Intercréanciers
# Pouvoirs à conférer en vue des formalités

# Ayant pris connaissance des conditions de financement de la centrale Neuillé-Pont-Pierre Sud et notamment
# des termes et conditions des projets du Contrat de Nantissement du Compte-Titres Financiers et de l’Accord
# Intercréanciers, le Conseil Communautaire décide à l’unanimité de :

#  Prendre acte que la conclusion et la mise en œuvre des opérations visées dans le Contrat de
# Nantissement du Compte-Titres Financiers et l’Accord Intercréanciers auxquels la Communauté de
# Communes Gâtine-Racan est partie, sont bien conformes à l'intérêt social du syndicat ;
#  Approuver les termes du projet de Contrat de Nantissement du Compte-Titres Financiers et du projet
# de l’Accord Intercréanciers auxquels la Communauté de Communes Gâtine-Racan est partie ;
#  Autoriser la signature et l’exécution du Contrat de Nantissement du Compte-Titres Financiers et de
# l’Accord Intercréanciers auxquels la Communauté de Communes Gâtine-Racan est partie, ainsi que
# tous autres documents devant être négociés et signés dans le cadre de la conclusion de la Convention
# de Crédit ou plus généralement en relation avec la Convention de Crédit ;
#  Autoriser le président avec possibilité de subdélégation, à signer le Contrat de Nantissement du
# Compte-Titres Financiers, l’Accord Intercréanciers et tout autre document devant être négocié et
# signé dans le cadre de la conclusion de la Convention de Crédit ou plus généralement en relation avec

# 10

# la Convention de Crédit et à effectuer toute déclaration, certification, formalité et démarche
# nécessaire ou utile à la conclusion des Documents de Financement afin de leur donner plein effet.

# E - Abattoir de Bourgueil – Demande de participation
# CC08_2023 ACTION ECONOMIQUE - ABATTOIR DE BOURGUEIL - Demande de
# participation
# Monsieur le Président donne les éléments suivants :
# La SCIC ABS, Société Coopérative d’Intérêt Collectif Abattoir Bourgueillois Services, a été mise en
# liquidation judiciaire le 28 juin 2022. Elle gérait l’activité de l’abattoir de Bourgueil.
# Lors d’une réunion de bilan de l’abattoir de Bourgueil du 11 juillet 2022 en présence du Conseil
# Régional, du Conseil Départemental, de Tours Métropole Val de Loire, des Communautés de
# Communes du Département d’Indre-et-Loire et de Janick Quentin, ancien Président de la SCIC ABS, il
# a été décidé collectivement de se donner les moyens de retrouver un repreneur à cet outil structurant
# pour les filières courtes des départements de l’Indre-et-Loire et du Maine-et-Loire.
# Afin de ne pas démanteler l’outil de production, il a été décidé, en l’absence de repreneur privé à
# l’issue de l’appel d’offre, que la Chambre d’Agriculture d’Indre-et-Loire se porte acquéreur de la
# chaîne d’abattage. Il a également été décidé que les collectivités participeraient au financement du
# rachat de la chaîne d’abattage.
# La valorisation de l’ensemble de l’actif de la SCIC ABS est de :
# -

# Valeur d’exploitation : 188 350 €
# Valeur de réalisation : 78 860 €

# Ci-dessous le tableau avec proposition de financement de reprise de la chaîne d’abattage qui détaille un
# projet de participation de tous les partenaires :

# REPRISE DE L’ACTIF – SCIC ABS
# PROPOSITION DE REPARTITION DE FINANCEMENT – HYPOTHESE
# RACHAT A 100 000 €
# DEPENSES
# POSTE DE
# DEPENSE
# CHAÎNE
# D’ABATTAGE

# RECETTES

# MONTANT
# 100 000 €

# FINANCEURS
# CHAMBRE
# D’AGRICULTURE
# CONSEIL REGIONAL
# CONSEIL
# DEPARTEMENTAL
# TOURS METROPOLE
# COMMUNAUTE DE
# COMMUNES GATINE –
# RACAN / CHINON
# VIENNE ET LOIRE /
# TOURAINE VALLEE DE
# L’INDRE / TOURAINE
# VAL DE VIENNE *
# COMMUNAUTE DE
# COMMUNES

# MONTANT

# TAUX %

# 15 000 €

# 15 %

# 15 000 €
# 15 000 €

# 15 %

# 15 000 €

# 15 %

# 12 000 €

# 12 %

# 5 000 €

# 5%

# 15 %

# 11

# CASTELRENAUDAIS / VAL
# D’AMBOISE / TOURAINE
# EST VALLEE / AUTOUR DE
# CHENONCEAU / LOCHES
# SUD TOURAINE **
# AUTOFINANCEMENT DE
# LA CHAMBRE
# 23 000 €
# 23 %
# D’AGRICULTURE
# COMMUNAUTE DE
# COMMUNES TOURAINE
# 0€
# 0%
# OUEST VAL DE LOIRE***
# TOTAL
# 100 000 €
# TOTAL
# 100 000 €
# 100 %
# * 3 000 € de participation pour chaque communauté de communes
# ** 1 000 € de participation pour chaque communauté de communes
# *** Proposition de non-participation au regard des fonds déjà investis dans la gestion immobilière du
# projet (déficit de 928 357 €)
# Pour la Communauté de Communes Gâtine – Racan et les Communautés de Communes Chinon
# Vienne et Loire, Touraine Vallée de l’Indre et Touraine Val de Vienne, le taux d’engagement proposé
# est de 12%, soit un montant de 12 000 € pour une hypothèse de rachat à 100 000 €, soit pour la
# Communauté de Communes Gâtine – Racan une participation de 3 000 €.
# Le montage juridique qui permettra de rétrocéder aux partenaires leur participation au prorata de la
# valeur de vente de la chaîne d’abattage est à l’étude.
# Le Conseil Communautaire à l’unanimité décide :
# -

# -

# de valider la participation financière de la Communauté de Communes Gâtine – Racan à la
# reprise de la chaîne d’abattage de l’Abattoir de Bourgueil,
# de valider ou non la proposition de taux d’engagement de 12 % pour la Communauté de
# Communes Gâtine – Racan et les Communautés de Communes Chinon Vienne et Loire,
# Touraine Vallée de l’Indre et Touraine Val de Vienne.
# d’autoriser, Monsieur le Président ou son représentant, à signer tous les documents afférents
# à ce dossier.

# M. Canon souligne que le bâtiment appartient à TOVAL.
# Intervention de M. Anceau : il est nécessaire de veiller à « bien manger dans nos collèges » …pas de
# repreneur… alors on achète !
# M. Verneau : TOVAL ne participe pas ? La Communauté de Communes TOVAL a acheté le bâtiment.
# Mme Plou s’étonne et pense que la Métropole et le Département ne font pas beaucoup d’efforts sur le
# sujet. Le Département a déjà participé en juin et en janvier et œuvre pour le développement
# économique.
# Les bêtes sont actuellement toutes découpées sur Vendôme.
# L’abattoir ne gagne pas d’argent ; il lui faut une plus-value en parallèle : pour cela il nous faut un
# repreneur.
# C’est un geste de solidarité Départementale quel que soit la couleur politique.
# M. Verneau demande s’il y a eu un audit sur la rentabilité de l’abattoir ; L’étude a déjà été faite.
# « Si pas de repreneur, on reprendra nos fonds ». On est en lien avec les projets « bien manger ».

# 12

# 4 – TOURISME
# A - Diners et Goûters du Patrimoine – Nouvelles modalités d’organisation
# CC10_2023 TOURISME - DINERS ET GOUTERS DU PATRIMOINE - NOUVELLE
# MODALITES D’ORGANISATION
# Monsieur le Président laisse la parole à Monsieur Canon qui indique que deux nouvelles modalités
# d’organisation sont à soumettre à l’avis du conseil communautaire :
# Il s’agit de prendre une délibération qui entérine ces choix (il ne s’agit plus d’option):
# Nouvelles modalités :
# Condition 1 : « Gestion de l’événement par la Communauté de Communes pour les nouveaux
# participants uniquement »
# Avec application des tarifs adoptés par la communauté de communes en 2023 :
# Pour rappel : plein (Adultes à partir de 18 ans) : 15 €
# Réduit : (Enfants de 12 à 17 ans, étudiants, demandeurs d’emploi, personnes bénéficiant du RSA,
# personnes en situation de handicap. Sur présentation d’un justificatif.) : 8 €
# De 6 à 11 ans : 5 €
# Jusqu’à 5 ans : gratuité
# Engagements du propriétaire :
# •Mettre gracieusement son site à disposition
# •Communiquer l’ensemble des informations nécessaires à la réalisation de la programmation (visuels,
# textes descriptifs, horaires, besoins en matériel)
# •Communiquer l’événement à ses réseaux
# •Préparer une visite guidée
# •Accompagner la logistique la veille et le jour de l’événement
# •Assurer son site pour l’événement (pas de changement de tarif)
# •Maintenir le lieu propre pour l’accueil
# •Participer à l’organisation de l’événement : accueil, parking, installation des chaises et de la collation
# •Participer à l’accueil du public et de la compagnie
# Engagements de la Communauté de Communes Gâtine-Racan :
# •Gérer l’organisation de l'événement
# •Gérer la relation avec la compagnie : devis et contrat de cession, restauration et hébergements des
# artistes
# •Effectuer les démarches administratives : déclaration Sacem, déclaration de l’événement en Mairie,
# demande autorisation débits de boissons si alcool, assurance responsabilité civile pour l’événement
# •Prendre en charge la communication : création et diffusion de la plaquette générale, relations presse,
# radio, encarts publicitaires, réseaux sociaux et signalétique
# •Préparer le dossier de subvention PACT et autres subventions possibles
# •Réserver le matériel auprès des communes et de la Communauté de Communes
# •Mettre en place et gérer le service de billetterie en ligne
# •Accompagner l’événement le jour J pour accueillir la compagnie et l’aider lors de l’installation
# •Accueillir le public et tenir la billetterie sur place pour les ventes de dernière minute, dans le cas où le
# spectacle ne serait pas complet par rapport à la jauge fixée
# •Collation fournie par la Communauté de Communes
# •Présence de l’Office de Tourisme de la Vallée du Loir (OTVL) sur les dates au Nord du territoire
# selon leurs possibilités
# 13

# Engagements de la commune :
# •Apporter et installer le matériel (tables, chaises, barnums, praticables…)
# •Installer les 3 banderoles dans la commune
# •Aide bénévole des élus pour l’organisation de l’événement
# •Participer à la communication
# Condition 2 : « Gestion de l’événement par le Propriétaire lorsqu’il a déjà participé à un Goûter
# et Dîner ou bien lorsqu’il organise déjà des événements »
# Tarifs 2023 :
# Au choix par le propriétaire
# Engagements du propriétaire :
# •Gérer l’organisation de l’événement
# •Effectuer ses démarches administratives : déclaration Sacem, déclaration de l’événement en Mairie,
# demande autorisation débits de boissons si alcool, assurance responsabilité civile pour l’événement
# •Réserver le matériel nécessaire auprès de sa commune ou tout autre prestataire
# •Préparer le dossier de subvention PACT
# •Gestion de la logistique avant, pendant et après l’événement
# •Assurer toutes les dépenses et prendre le risque financier : bénéfices ou pertes
# •Gérer la relation avec la compagnie : devis et contrat de cession, restauration et hébergements des
# artistes
# •Assurer les ventes de billets : gestion de sa billetterie en ligne, réservations et sur place le jour J dans
# le cas où le spectacle ne serait pas complet
# •Communiquer l’événement à ses réseaux
# •Préparer une visite guidée
# •Maintenir le lieu propre pour l’accueil
# •Accueillir le public et la compagnie
# •Communiquer les informations de l’événement à la Communauté de Communes pour la plaquette de
# programmation (visuels, textes, déroulé de l’événement, tarifs et moyens de réservation)
# •Possibilité d’une collation
# Engagements de la Communauté de Communes Gâtine-Racan :
# •Prendre en charge la communication : création et diffusion de la plaquette générale, relations presse,
# radio, encarts publicitaires, réseaux sociaux
# •Accompagner et déposer le dossier de subvention PACT (Service Culture)
# •Mise à disposition des banderoles et des panneaux signalétiques
# •Présence de l’Office de Tourisme de la Vallée du Loir (OTVL) sur les dates au Nord du territoire
# selon leurs possibilités
# Engagements de la commune :
# •Mettre à disposition le matériel si besoin
# •Installer les 3 banderoles dans la commune
# •Aide bénévole des élus disponibles selon les besoins
# •Participer à la communication
# Le Conseil Communautaire décide à l’unanimité de :
# - Valider la nouvelle organisation inhérente aux gouters et diners du patrimoine telle que
# proposée ci-dessus,
# - Autoriser, Monsieur le Président ou son représentant, à signer tous les documents afférents
# à ce dossier.

# 14

# B – Goûters et Dîners du Patrimoine – Définition des tarifs
# CC09_2023 TOURISME - DEFINITION DES TARIFS - GOUTERS ET DINERS DU
# PATRIMOINE
# Monsieur le Président présente les éléments suivants :
# La Communauté de Communes Gâtine-Racan possède un patrimoine architectural et historique
# remarquable. Ce patrimoine appartient en majorité à des propriétaires privés et beaucoup de ces
# sites ne sont pas ouverts à la visite.
# En 2019, la collectivité a donc souhaité fédérer plusieurs propriétaires autour d’un cycle
# d’événements en leur proposant d’ouvrir leurs portes durant l’été.
# Ce cycle d’événements met ainsi en lumière les sites du patrimoine de la Communauté de
# Communes Gâtine-Racan habituellement fermés au public associé à un spectacle d’une
# compagnie locale (théâtre, musique, art équestre…) suivi d’un goûter, l’objectif étant de
# valoriser ce patrimoine secret et de rendre accessible la culture en milieu rural au plus grand
# nombre.
# Des tarifs ont été votés par délibération le 8 décembre 2022, comme suit :
# - Tarif adulte à 15 €.
# •
# •
# •
# •
# •

# -

# Tarif réduit à 8 € comprenant :
# Enfants de 12 à 17 ans,
# Étudiants 18-25 ans,
# Demandeurs d’emploi,
# Personnes bénéficiant du RSA,
# Personnes en situation de handicap.
# Sur présentation d’un justificatif.
# - Gratuité pour les moins de 12 ans

# Monsieur le Président soumet au vote du conseil communautaire la mise en place d’un nouveau
# tarif pour les enfants de 6 à 11 ans à hauteur de 5 euros, et ce, uniquement dans le cadre d’une gestion
# de l’événement par la Communauté de Communes pour les nouveaux participants (Cf délibération
# CC10.2023 du 25.01.2023) (gratuité jusqu’à 5 ans)
# Entendu la présentation de Monsieur le Président,
# Le Conseil communautaire décide à l’unanimité de :
# -

# Valider les tarifs ci-dessus présentés avec notamment le nouveau pour les enfants de 6 à
# 11 ans à hauteur de 5 euros
# Autoriser, Monsieur le Président ou son représentant, à signer tous les documents
# afférents à ce dossier.

# C - Information Randonnées
# Retour sur le sujet : tout se passe bien. Laura rencontre toutes les Communes et nous sommes dans les
# temps.
# La carte sera plus importante que prévue au départ.
# 15

# La boucle de randonnée sera sur les panneaux…et on ne mettra que cela… c’est juste pour matérialiser
# le départ…il y aura un QR Code partout.

# 5 – RESSOURCES HUMAINES
# A – Création d’un emploi permanent - Assistant culture
# CC11_2023 RESSOURCES HUMAINES - CREATION EMPLOI PERMANENT ASSISTANT
# CULTURE
# Monsieur le Président expose les éléments suivants :
# Vu le Code général de la fonction publique, notamment les articles L.2, L.7 et L.332-8 2°,
# Vu la loi n° 82-213 du 2 mars 1982 modifiée relative aux droits et libertés des communes, des
# départements et des régions, notamment son article 1,
# Vu le décret n°88-145 du 15 février 1988 modifié, pris pour l'application de l'article 136 de la loi n° 8453 du 26 janvier 1984 modifiée portant dispositions statutaires relatives à la fonction publique
# territoriale et relatif aux agents contractuels de la fonction publique territoriale,
# Vu le budget,
# Vu le tableau des emplois et des effectifs,
# Conformément à l’article L313-1 du Code Général de la Fonction Publique, susvisé les emplois de
# chaque collectivité ou établissement sont créés par l’organe délibérant de la collectivité ou de
# l’établissement.
# Il appartient donc au Conseil Communautaire de fixer l’effectif des emplois nécessaires au
# fonctionnement des services.
# Considérant la nécessité de pérenniser le poste d’assistant culture pour assurer le bon fonctionnement
# des missions du service, la Communauté de Communes Gâtine-Racan souhaite créer un emploi
# permanent d’assistant culture, à temps complet, à compter du 26 janvier 2023.
# Cet emploi pourra être pourvu par un fonctionnaire de catégorie C de la filière administrative, du cadre
# d’emploi des Adjoints Administratifs Territoriaux.
# Au regard de la spécificité de l’emploi, de l’expertise et des compétences attendues, et si le recrutement
# d’un fonctionnaire s’avère infructueux, l’emploi pourra être occupé par un agent contractuel relevant de
# la catégorie B conformément à l’article L.332-8 2° du Code général de la fonction publique qui permet
# aux collectivités territoriales et aux établissements publics locaux lorsque les besoins des services ou la
# nature des fonctions le justifient et sous réserve qu'aucun fonctionnaire territorial n'ait pu être recruté
# dans les conditions prévues par le Code général de la fonction publique, de recruter un contractuel sur
# tout emploi permanent.
# L’agent contractuel sera alors recruté par voie de contrat à durée déterminée pour une durée de 3 ans.
# Le recrutement de l’agent contractuel sera prononcé à l’issue d’une procédure prévue par les décrets
# n°2019-1414 du 19 décembre 2019 et n°88-145 du 15 février 1988, ceci afin de garantir l’égal accès
# aux emplois publics.
# Ce contrat sera renouvelable par reconduction expresse en respectant la procédure de recrutement
# mentionnée ci-dessus. La durée totale des contrats ne pourra excéder 6 ans. A l’issue de cette période
# maximale de 6 ans, le contrat de l’agent sera reconduit pour une durée indéterminée.
# L’agent contractuel devra justifier d’une expérience professionnelle dans le secteur du spectacle vivant
# et de la gestion administrative.
# Sa rémunération sera calculée par référence à l’échelle indiciaire du grade de rédacteur territorial du
# cadre d’emplois des rédacteurs territoriaux.
# La rémunération sera déterminée en prenant en compte, notamment, les fonctions occupées, la
# qualification requise pour leur exercice, la qualification détenue par l’agent contractuel ainsi que son
# expérience.
# 16

# Le Conseil Communautaire à l’unanimité décide de :
# -

# -

# La création d’un emploi permanent d’assistant culture, à temps complet, à compter du 26
# janvier 2023, relevant de la catégorie hiérarchique C, de la filière administrative, du cadre
# d’emploi des Adjoints administratifs territoriaux ;
# De modifier en conséquence le tableau des emplois et des effectifs et de remplacer le poste
# 2F, emploi non permanent par un emploi permanent ;
# D’autoriser, dans l’hypothèse du recrutement infructueux d’un fonctionnaire et en raison
# des besoins du service ou de la nature des fonctions, Monsieur le Président à recruter un
# agent contractuel sur le fondement de l’article L.332-8 2° du Code général de la fonction
# publique et à signer le contrat afférent ;
# De préciser que ce contrat sera d’une durée initiale de 3 ans renouvelable expressément ;
# De préciser que la rémunération sera fixée en référence à l’échelle indiciaire du grade
# rédacteur du cadre d’emplois des Rédacteurs Territoriaux.

# M. le Président précise que les crédits nécessaires sont inscrits au budget de la Communauté de
# Communes.

# B – Création d’un emploi non permanent - Accroissement saisonnier d’activité Animation
# CC12_2023 RESSOURCES HUMAINES - CREATION EMPLOI NON PERMANENT ACCROISSEMENT SAISONNIER ANIMATION
# Monsieur le Président expose les éléments suivants :
# Vu le Code Général de la Fonction Publique, notamment son article L332-23-2° ;
# Considérant qu’il est nécessaire de recruter 1 agent contractuel pour faire face à un besoin lié à un
# accroissement saisonnier d’activité,
# Le Conseil Communautaire décide à l’unanimité de :
# -La création au tableau des effectifs, à compter du 26 janvier 2023 d’un emploi non permanent pour
# faire face à un besoin lié à un accroissement saisonnier d’activité dans le grade d’adjoint
# d’animation, relevant de la catégorie hiérarchique C, à temps complet, pour une durée
# hebdomadaire de 35 heures ;
# -Cet emploi non permanent sera occupé par un agent contractuel recruté par voie de contrat à durée
# déterminée pour une durée de 6 mois maximum pendant une même période de 12 mois.
# -L’imputation des dépenses correspondantes sur les crédits prévus à cet effet au budget.
# M. le Président informe l’assemblée délibérante du départ de Clémence, récemment arrivée sur le poste
# des marchés publics.

# 6 - ECHANGES ENTRE ELUS
# M. le Président donne lecture de la note préparée par Valérie et distribuée sur table concernant le
# COTECH et les différentes étapes inhérentes à l’avancée de notre procédure sur le PLUI.

# 17

# DEBAT SUR LES ORIENTATIONS BUDGETAIRES :
# M. le Président explique que cette présentation n’est pas obligatoire.
# Les orientations présentent les projets en dépenses et en recettes, la gestion de la structure et également
# le volet inhérent aux Ressources Humaines.
# M. le Président remercie Bertrand et Muriel, venue en renfort pour le travail accompli.
# Il indique également avoir une pensée pour Mme Percereau, actuellement en arrêt maladie et qui
# affectionnait particulièrement le domaine des Finances.
# M. le Président : « Nous sortons de deux années de crise sanitaire, nous sommes au cœur d’une autre
# crise, celle-ci économique, et qui fait suite à un conflit, l’inflation en progression et notamment la
# problématique énergétique notoire, les positions du gouvernement fluctuantes…les citoyens sont capés
# à 15 % d’augmentation mais pas notre collectivité ».
# Le projet de loi de finances 2023 se base sur 2.6 du PIB avec une décélération possible pour 2023.
# M. le Président annonce quelques chiffres clés : hypothèse de croissance sur l’année 2023 prévue à 1%
# voire moins selon certains organismes.
# Le déficit public s’établirait en 2023 à 5% du PIB et la revalorisation forfaitaire des valeurs locatives
# qui pourrait atteindre +7.1 %.
# L’évolution du montant perçu pour la fraction de TVA en 2023 pourrait s’élever à 5% par rapport à
# 2022.
# M. le Président indique qu’on nous promet la stabilité des dotations et une réforme minimale des
# indicateurs financiers.
# Un soutien à l’investissement local : avec la création d’un fond vert doté de 2 milliards d’euros.
# Notre Coefficient d’intégration fiscale est supérieur à 0.50. (Un des meilleurs des collectivités d’Indre
# et Loire) ce qui permettra de ne pas engendrer de baisse sur le calcul de la dotation globale de
# fonctionnement.
# Il est à noter l’augmentation importante des coûts d’énergie : le groupement de commande avec le
# syndicat d’énergie permettrait de limiter les hausses.
# Sur ce sujet il n’y a pas d’équité entre les petites Communes et les plus importantes.
# M. le Président souligne qu’au regard du contexte, beaucoup de Communes font désormais la chasse
# aux dépenses et sont particulièrement attentives aux dépenses liées à l’énergie.
# Les orientations budgétaires ont été construites sans augmentation des impôts : il est annoncé des
# augmentations des bases ce qui provoquera inévitablement des augmentations pour les entreprises et les
# citoyens.
# Sur la taxe d’habitation on passera à 2 251 200 euros.
# M. le Président présente le sujet des attributions de compensation : elles sont librement révisées chaque
# année : CLECT avec reversement des Communes sauf pour le PLUI.
# Reprise de la compétence Transport : il faudra se poser sérieusement la question.
# Le FPIC : 602903 euros en 2022…600 000 prévue en 2023 (nous sommes sur des masses). La loi de
# finances en prévoie la stabilité Nationale.
# Concours de l’état : en 2021 nous étions en dotation d’intercommunalité à 753 860 euros et 2022 est à
# 760 146 euros
# M. le Président souligne que notre territoire est attractif et nous avons désormais 22 206 habitants.
# Concernant les recettes (produits des services).
# 18

# Deux ventes de terrains sont certaines pour ce qui est du développement économique. Nous restons
# dans l’attente du dossier CATELLA : Le protocole a été rédigé pour trouver un accord.
# Concernant la fiscalité sur le budget des ordures ménagères : le produit de la TEOM est de 2 282 000
# euros et 151 000 pour la redevance spéciale.
# M. le Président donne lecture des chiffres en Ressources Humaines et explique l’évolution pour
# l’exercice 2023.
# Charges du personnel : en progression avec 3.5 d’indice appliqué notamment et 739 160 euros reversés
# aux Communes (en lien avec la compétence Voirie). Cela représente 26 % des dépenses réelles.
# Il y a désormais 50 personnes dans la structure : est-ce que cela change quelque chose ? Il faut résonner
# sur des ETP.
# L’épargne brute est d’environ 1.9 million.
# La capacité de désendettement de la collectivité est de 6 ans.
# On a deux gros projets cette année : 1 million HT chacun.
# M. le Président donne lecture des chiffres inhérents aux dépenses d’investissement.
# Les réalisations : réhabilitation énergétique avec le changement de toutes les fenêtres.
# La Communauté de Communes a terminé le co-working. On a fait la plateforme Tri’Tout de StPaterne.
# Au 1er janvier 2024 la loi interdira les fermenticides dans le sac noir. Il faut encourager à l’utilisation
# des composteurs.
# M. le Président indique avoir rencontré tous les Vice-Présidents et que seules les « nouveautés » sont
# actées dans le document. Il est procédé à la lecture des documents en fonction de toutes les matières et
# les services.
# Sur le PAT / accompagnement des maraichers.
# Environnement : projet d’arbres pour le territoire.
# Il est ici précisé qu’il faut laisser les commissions formuler des propositions.
# Il nous faut une personne pour le suivi de collecte et il y a trop d’erreur sur le sac noir. Les erreurs de
# tri nous coûtent très cher.
# Le Budget Général devra abonder le budget OM de 300 000 euros.
# Environnement : Si nous passons en C05 : il faut acheter du matériel à hauteur de 300 000 euros.
# Il est proposé de nouvelles filières : achats de plots béton etc…. broyeur à végétaux…
# Eco pâturage sur la salle des 4 vents.
# En Culture : matériel en régie, souci de chauffage avec une géothermie qui dysfonctionne, console pour
# les lumières (on a bloqué le devis pour bloquer les tarifs).
# Eclairage en salle arrière et de la signalétique.
# Bâtiment : Part importante pour le multi-accueil.
# ADAP : accès aux personnes à mobilité réduite de tous nos bâtiments.
# Renouveler tous les éclairages intérieurs.
# Sur le sujet de la Voirie : application du principe « un pour un » : les petites Communes sont
# pénalisées. Une proposition est faite avec fonds de concours pris sur le budget de la Communauté de
# Communes pour les Communes de -1000 habitants.
# -8000 euros par Commune et par an / jusqu’à la fin du mandat.
# Ex : Une Commune peut vouloir utiliser 16 000 euros sur une année et rien sur l’autre année etc…
# M. le Président balaye les propositions en développement économique : en dehors des aménagements
# des zones …signalétique sécurisation etc…
# En tourisme : plaquettes etc…
# Terrain de foot synthétique sur NPP (36 000 de MO…).
# Ces projets sont le résultat des rendez-vous que Messieurs Trystram et Peninon ont tenu.
# Il est souligné la volonté de stabiliser la pression fiscale.
# M. le Président demande à tous les Vice-Présidents d’être vigilants pour contenir les dépenses.
# M. Verneau félicite les services pour la fourniture du lexique.
# 19

# Monsieur le Président donne les informations réglementaires suivantes :
# Le code général des collectivités territoriales précise, dans son nouvel article L2312-1 (modifié par LOI
# n° 2015-991 du 7 août 2015 - art. 107) que dans les communes de 3 500 habitants et plus, le Maire
# présente au Conseil municipal, dans un délai de deux mois précédant l'examen du budget, un rapport
# sur les orientations budgétaires (ROB), les engagements pluriannuels envisagés ainsi que sur la
# structure et la gestion de la dette,
# Dans l'article L.2121-8, le ROB donne lieu à un débat au Conseil municipal, dans les conditions fixées
# par le règlement intérieur et qu'il est pris acte de ce débat par une délibération spécifique,
# Considérant que pour notre collectivité qui ne possède pas de communes de plus de 3500 habitants, ce
# débat n’est pas une obligation réglementaire,
# Considérant cependant que le débat d'orientation budgétaire constitue une étape importante dans le
# cycle budgétaire annuel d'une collectivité locale : si l'action d'une collectivité est principalement
# conditionnée par le vote du budget primitif, le cycle budgétaire est rythmé par la prise de nombreuses
# décisions et ce débat permet à l'assemblée de discuter des orientations budgétaires qui préfigurent les
# priorités qui seront inscrites dans le cadre du budget primitif,
# Le Conseil Communautaire :
# - Prend acte de la tenue du débat d'orientations budgétaires 2023 et du rapport du débat
# d’orientations budgétaires qui sera annexé à la délibération, et dit que la présente délibération sera
# transmise au contrôle de légalité.
# Information : Pour le PLUI
# 18 avril 2023 : la Commune de Sonzay met à disposition sa salle des associations pour la réunion et la
# tenue des ateliers pour le diagnostic foncier - à voir avec le cabinet CITTANOVA.
# M. Desjonquères s’interroge sur le Transport : emplois de la Communauté de Communes afférents à
# ces missions ? Maintien de la compétence ?
# On avait choisi de confier la compétence Transport à la Région.
# M. Canon : il signale qu’il est important de conserver ce service. Il est important de statuer sur le sujet
# assez rapidement. Il faut réfléchir également sur la CLECT…
# M. le Président signale que Mme Percereau sera absente jusqu’à la mi-février.
# Il conviendra de régler le problème de la gestion de sa boite mail.
# Intervention de M. Capon :
# Concernant la deuxième édition du Budget Participatif du Département il dénonce l'impossibilité pour
# les petites communes de voir un seul de leur projet élu vu le mode d'élection retenu ! "
# Levée de séance : 21h30

# 20








#  请直接给出答案,不要做任何解释。
# """