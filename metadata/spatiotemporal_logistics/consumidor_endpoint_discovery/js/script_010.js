function Mascara(componente,tipo_mascara){
	
    v_obj=componente;
    v_fun=tipo_mascara;

    setTimeout("execmascara()",1);
}

/*Funcao que executa os objetos*/
function execmascara(){
    v_obj.value=v_fun(v_obj.value);
}

/*Funcao que Determina as expressoes regulares dos objetos*/
function leech(v){
    v=v.replace(/o/gi,"0");
    v=v.replace(/i/gi,"1");
    v=v.replace(/z/gi,"2");
    v=v.replace(/e/gi,"3");
    v=v.replace(/a/gi,"4");
    v=v.replace(/s/gi,"5");
    v=v.replace(/t/gi,"7");
    return v;
}

/*Funcao que permite apenas numeros*/
function Integer(v){
    return v.replace(/\D/g,"");
}

/* Funcao que permite numeros e letras. */
function NumeroLetras(v){
    return v.replace(/[^a-zA-Z0-9]/g,"");
}

/* Login com letras (sem acento), numeros e underline */
function Login(v){
    return v.replace(/[^a-z0-9_\.]/g,"");
}

/* Funcao permitindo apenas numeros e hifen */
function NumeroHifen(v){
    return v.replace(/[^0-9-]/g,"");
}

/*Funcao que padroniza telefone (11) 4184-1241 */
function Telefone(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/^(\d\d)(\d)/g,"($1) $2");
    v=v.replace(/(\d{4,5})(\d{4})/,"$1-$2");
    return v;
}

/*Funcao que padroniza telefone (11) 41841241*/
function TelefoneCall(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/^(\d\d)(\d)/g,"($1) $2");
    return v;
}

/*Funcao que padroniza CPF*/
function Cpf(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/(\d{3})(\d)/,"$1.$2");
    v=v.replace(/(\d{3})(\d)/,"$1.$2");
    v=v.replace(/(\d{3})(\d{1,2})$/,"$1-$2");
    return v;
}

/*Funcao que padroniza o cÃ³digo CNAE 1111-1/11 */
function Cnae(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/(\d{4})(\d)/, "$1-$2");
    v=v.replace(/(\d{1})(\d{1,2})$/,"$1/$2");
    return v;
}

/*Funcao que padroniza CEP*/
function Cep(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/^(\d{5})(\d)/,"$1-$2");
    return v;
}

/*Funcao que padroniza CNPJ*/
function Cnpj(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/^(\d{2})(\d)/,"$1.$2");
    v=v.replace(/^(\d{2})\.(\d{3})(\d)/,"$1.$2.$3");
    v=v.replace(/\.(\d{3})(\d)/,".$1/$2");
    v=v.replace(/(\d{4})(\d)/,"$1-$2");
    return v;
}

/*Faz a mascara de cpf ou login */
function CpfOuLogin(v){
	var aux = v; 
    aux=aux.replace(/\./g,"");
    aux=aux.replace(/\//g,"");
    aux=aux.replace(/\-/g,"");

    if (!isNaN(aux)) {
    	v=Cpf(aux);
    }
    return v;
}

/* Faz a mascara de cnpj ou cpf dependendo do tamanho dos digitos */
function CpfCnpjContagem(v){
	v=v.replace(/\D/g,"");
    v=v.replace(/\./g,"");
    v=v.replace(/\//g,"");
    v=v.replace(/\-/g,"");

    if(v.length == 11){
        v=Cpf(v);
    }else if(v.length == 14){
        v=Cnpj(v);
    }
    
    return v;
}

/* Funcao que padroniza Protocolo */
function Protocolo(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/^(\d{4})(\d)/,"$1.$2");
    v=v.replace(/^(\d{4})\.(\d{2})(\d)/,"$1.$2/$3");
    v=v.replace(/(\d{11})(\d)/,"$1");
	return v;
}

/*Funcao que permite apenas numeros Romanos*/
function Romanos(v){
    v=v.toUpperCase();
    v=v.replace(/[^IVXLCDM]/g,"");

    while(v.replace(/^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$/,"")!="")
        v=v.replace(/.$/,"");
    return v;
}

/*Funcao que padroniza o Site*/
function  Site(v){
    v=v.replace(/^http:\/\/?/,"");
    dominio=v;
    caminho="";
    if(v.indexOf("/")>-1)
        dominio=v.split("/")[0];
    caminho=v.replace(/[^\/]*/,"");
    dominio=dominio.replace(/[^\w\.\+-:@]/g,"");
    caminho=caminho.replace(/[^\w\d\+-@:\?&=%\(\)\.]/g,"");
    caminho=caminho.replace(/([\?&])=/,"$1");
    if(caminho!="")dominio=dominio.replace(/\.+$/,"");
    v="http://"+dominio+caminho;
    return v;
}

/*Funcao que padroniza DATA*/
function Data(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/(\d{2})(\d)/,"$1/$2");
    v=v.replace(/(\d{2})(\d)/,"$1/$2");
    return v;
}

/*Funcao que padroniza DATA no formato MM/YYYY*/
function DataMesAno(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/(\d{2})(\d)/,"$1/$2");
    return v;
}

/*Funcao que padroniza HORA*/
function Hora(v){
    v=v.replace(/\D/g,"");
    v=v.replace(/(\d{2})(\d)/,"$1:$2");
    return v;
}

function HoraFull(v){
	v=v.replace(/\D/g,"");
	v=v.replace(/(\d{2})(\d)/,"$1:$2");
	v=v.replace(/(\d{2})(\d)/,"$1:$2");
	return v;
}

/*Funcao que padroniza valor monetario*/
function Valor(v){
    v=v.replace(/\D/g,""); //Remove tudo o que nao eh digito
    v=v.replace(/^([0-9]{3}\.?){3}-[0-9]{2}$/,"$1.$2");
    v=v.replace(/(\d)(\d{2})$/,"$1,$2"); //Coloca ponto antes dos 2 ultimos digitos
    v=v.replace(/(\d{1,3})(\d{3})([.,].*)/,"$1.$2$3");
    v=v.replace(/(\d{1,3})(\d{3})([.,].*)/,"$1.$2$3");
    return v;
}

/*Funcao que padroniza valor de indices*/
function Indice(v){
    v=v.replace(/\D/g,""); //Remove tudo o que nao eh digito
    v=v.replace(/^([0-9]{3}\.?){3}-[0-9]{2}$/,"$1.$2");
    v=v.replace(/(\d)(\d{4})$/,"$1,$2"); //Coloca virgula antes dos 4 ultimos digitos
    return v;
}

/*Funcao que padroniza Area*/
function Area(v) {
    v=v.replace(/\D/g,"");
    v=v.replace(/(\d)(\d{2})$/,"$1.$2");
    return v;

}

function CaracteresSimples(v) {
    v=v.replace(/[^a-zA-Z0-9\-+\(\)\*\?\!&%$#@\.,:"'\s]/g,"");
    return v;
}

function Letras(v) {
	v = v.replace(/[\W\d]/g, "");
	return v;
}

function LetrasMaiusculas(v) {
	v = Letras(v);
	return v.toUpperCase();
}

/**
 * Caracteres simples cem acento (nÃ£o permite invÃ¡lidos)
 */
function LetrasComAcento(v) {
	v=v.replace(/[^A-Za-zÃ-Ã¼ ]/g,"");
	return v;
}

function LetrasENumeros(v) {
	v=v.replace(/[^A-Za-zÃ-Ã¼0-9 ]/g,"");
	return v;
}

function LetrasENumerosEAspas(v) {
	v=v.replace(/[^A-Za-zÃ-Ã¼0-9\" ]/g,"");
	return v;
}

/**
 * Funcao que mascara os campos que terÃ£o nome de credenciada. Permite letras, numero e parenteses.
 * 
 * @param v
 * @returns
 */
function NomeCredenciada(v) {
	v=v.replace(/[^A-Za-zÃ-Ã¼0-9\(\) ]/g,"");
	return v;
}

/*Verificando se a string passada pelo parametro eh uma e-mail (isEmail).*/ 
function isEmail(string){
    emailRegExp = /^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.([a-z]){2,4})$/;
    return emailRegExp.test(string);
}

/*Verificando se a string passada pelo parametro eh numerico IP (isIP).*/
function isIPv4(ip){
    ipRegExp = /^(([0-2]*[0-9]+[0-9]+)\.([0-2]*[0-9]+[0-9]+)\.([0-2]*[0-9]+[0-9]+)\.([0-2]*[0-9]+[0-9]+))$/;
    return ipRegExp.test(ip);
}

function CartaoCredito(v) {
    v=v.replace(/\D/g,"");
    v=v.replace(/^(\d{19}).*$/, "$1");
    v=v.replace(/^(\d{4})(\d{4})(\d{4})(\d{4})$/,"$1-$2-$3-$4");
    return v;
}

/**
 * Completa com zeros a quantidade indicada. Somente para numeros.
 * 
 * Numero: o valor a ser parseado.
 * Tamanho: quantidade de caracteres a serem preenchidos a esquerda
 * Padding: qual o caractere sera preenchido. Caso nao passe nada, por default sera '0'.
 */
function CompletarNumerosEsquerda(numero, tamanho, padding){
   var sign = '', resultado = numero;
 
   if (typeof numero === 'number'){
      sign = numero < 0 ? '-' : '';
      resultado = Math.abs(numero).toString();
   }
 
   if ((tamanho -= resultado.length) > 0){
	   resultado = Array(tamanho + 1).join (padding || '0') + resultado;
   }
   return sign + resultado;
}