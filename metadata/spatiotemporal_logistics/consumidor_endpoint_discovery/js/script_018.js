var abriuModalMsgTempoSessao = false;

$(document).ready(function () {
    // Placeholder para ie8 e 9
    $('input, textarea').placeholder();

    // Aciona o botÃ£o enter par ao formulario de login
    $("#loginPageForm input").keypress(function (event) {
        if (event.which == 13) {
            event.preventDefault();
            $("#loginPageForm").submit();
        }
    });
    // DataTable nÃ£o retira o display none quando executa o sort, esta funÃ§Ã£o
    // remove o display none quando clica na th da DataTable, assim a funÃ§Ã£o de
    // processing Ã© executada sem problema
    $('.dataTable th').click(function () {
        $('.dataTables_processing').css('display', '');
    });

    // $.getScript(window.location.protocol + '//barra.brasil.gov.br/barra.js');
});

function avaliarTempoSessao(){
    console.log('--> avaliarTempoSessao...')
    $.get( pageContext+"/pages/principal/timeout-sessao.json", function(respostaSessao){
        if(respostaSessao && respostaSessao != null && respostaSessao != undefined){
            console.log('respostaSessao', respostaSessao)
            if(respostaSessao.logado != null && respostaSessao.logado != undefined && (respostaSessao.logado === false || respostaSessao.logado === "false")){
                console.log('respostaSessao.logado', respostaSessao.logado)
                exibirMensagemErro("Seu tempo logado acabou. Qualquer aÃ§Ã£o redirecionarÃ¡ para fora da Ã¡rea logada. FaÃ§a cÃ³pia de textos digitados ainda nÃ£o enviados.");
            }
        }
    });
    setTimeout(avaliarTempoSessao, 15000);
}
