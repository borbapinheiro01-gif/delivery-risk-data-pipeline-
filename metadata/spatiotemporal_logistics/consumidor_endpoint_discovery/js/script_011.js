// ----------------------------------------
// Carregamentos iniciais
// ----------------------------------------
$.ajaxSetup({
    cache: false
});

/* Desabilita o keydown da tecla ESC no Internet Explorer para que ao digitar Esc 2x os formularios nao sejam limpos */
$(document).ready(function() {
    $(document).keydown(function(event) {
        if (navigator.appName == "Microsoft Internet Explorer" && event.keyCode == 27) {
            return false;
        }
        return true;
    });
});

$.urlParam = function(name) {
    var results = new RegExp('[\\?&]' + name + '=([^&#]*)').exec(window.location.href);
    if (results == null) {
        return null;
    }
    else {
        return results[1] || 0;
    }
}
// ----------------------------------------
function limitMaxlength(comp, event, lim) {
    event = event || window.event;
    if (event.keyCode != 8) {
        var str = comp.value;
        var length;
        if (!str){
            length = 0;
        }
        var lns = str.match(/(\r\n|\n|\r)/gm);
        if (lns){
            length = str.length + lns.length;
        }else{
            length = str.length;
        }
        if (length >= lim) {
            return false;
        }
    }
}
function retirarMenuLogado() {
    $(document).ready(function() {
        $("#sidebar").hide();
        $("#conteudo-decorator").prop("class", "col-lg-12 col-md-12 col-sm-12 col-xs-12 no-padding pull-right");
    });
}

function tratarEnter(event) {
    event = event || window.event;
    if (event.keyCode == 13) {
        buscarEmpresas();
    }
}

function limparErrosForm(idForm) {
    $(':input, :checkbox, :radio, select', "#" + idForm)
            .not(':button, :submit, :reset')
            .each(function(index, element) {

                apagaErro(element);
                apagaErroCustom(element);

                $(element).parent().removeClass("has-error");

                //APAGA OS CAMPOS DE ERROS ESPECIFICOS
                $('.msg-erros-valid').each(function(index, element) {
                    $('#' + element.id).text('');
                });
            });
}

function limparCamposForm(idForm, negar) {
    $(':input', "#" + idForm)
            .not(':button, :submit, :reset, :hidden, :radio, :checkbox, '+negar)
            .val('');

    $(':radio, :checkbox', "#" + idForm)
            .removeAttr('checked')
            .removeAttr('selected');

    limparErrosForm(idForm);
}

function mostraErro(el, msg) {
    var id = $(el).attr('id') + ".custom.errors";
    if (id && id != null) {
        $(el).after("<span id='" + id + "' class='has-error'>" + msg + "</span>");
        $("#" + id.replace(/\./g, "\\.")).fadeOut(0).fadeIn(100);
    }
}

function apagaErroCustom(element) {
    var id = $(element).attr('id') + ".custom.errors";
    if (id && id != null) {
        $("#" + id.replace(/\./g, "\\.")).remove();
    }
}

function apagaMostraErroCustom(el, msg) {
    apagaErroCustom(el);
    mostraErro(el, msg);
}

function apagaErro(element) {
    var id = $(element).attr('id');
    if (id && id != null) {
        if (id && id.indexOf('cmb') > -1) {
            id = id.substring(3).toLowerCase();
        }
        var comp = $("#" + id.replace(/\./g, "\\.") + "\\.errors");
        if (comp) {
            comp.remove();
        }
    }
}

function desabilitarColar(el, event) {
    event.preventDefault();
    return false;
}

function verificaConfirmacao(campo, confirmacao, toLower) {
    apagaErroCustom(confirmacao);
    var valorCampo = $(campo).val();
    var valorConfirmacao = $(confirmacao).val();

    if (valorConfirmacao && valorCampo) {

        if (toLower) {
            valorCampo = valorCampo.trim().toLowerCase();
            valorConfirmacao = valorConfirmacao.trim().toLowerCase();
        }

        if (valorConfirmacao !== valorCampo) {
            mostraErro(confirmacao, "Confirma&ccedil;&atilde;o inv&aacute;lida.");
        }
    }
}

function obtemEndereco(cpfEl, reloaded) {
    if (!$(cpfEl).val() || $(cpfEl).val().length < 9) {
        limparEndereco();
        apagaMostraErroCustom(cpfEl, "CEP inv\u00e1lido.");
        return false;
    }

    $.ajax({
        type: "GET",
        url: pageContext + '/pages/cadastroCep/' + $(cpfEl).val() + '/buscar.json',
        dataType: "json",
        context: this,
        statusCode: {
            404: function() {
                limparEndereco();
                apagaMostraErroCustom(cpfEl, "CEP n\u00E3o encontrado.");
            },
            403: function() {
                limparEndereco();
                apagaMostraErroCustom(cpfEl, "CEP inv\u00e1lido.");
            },
            500: function() {
                limparEndereco();
                apagaMostraErroCustom(cpfEl, "Falha na busca do CEP.");
            }
        },
        success: function(data) {
            if (data['logradouro']) {
                $("#logradouro").val(data['logradouro']);
                $("#logradouro").attr('readonly', 'readonly');
            } else {
                if (!reloaded) {
                    $("#logradouro").val('');
                }
                $("#logradouro").removeAttr('readonly');
            }
            if (data['bairro']) {
                $("#bairro").val(data['bairro']);
                $("#bairro").attr('readonly', 'readonly');
            } else {
                if (!reloaded) {
                    $("#bairro").val('');
                }
                $("#bairro").removeAttr('readonly');
            }
            $("#cidade").val(data['localidade']);
            $("#uf").val(data['uf']);
            $("#uf").trigger("change");
            apagaErroCustom(cpfEl);
        }
    });
}

function abrirModalBuscaCep() {
    var urlRequisicao = pageContext + "/pages/cadastroCep/busca/abrir";

    $.get(urlRequisicao, function(data, textStatus, jqXHR) {
        data = data.replace("<meta name=\"decorator\" content=\"application_modal\" />", "");
        $("#dlgBuscaCep").html(data);
    }).done(function() {
        $("#dlgBuscaCep").modal('show');
    });

    return false;
}

function abrirModalBuscaCnae() {
    var urlRequisicao = pageContext + "/pages/cnae/busca/abrir";

    $.get(urlRequisicao, function(data, textStatus, jqXHR) {
        data = data.replace("<meta name=\"decorator\" content=\"application_modal\" />", "");
        $("#dlgBuscaCnae").html(data);
    }).done(function() {
        $("#dlgBuscaCnae").modal('show');
    });

    return false;
}

function abrirModalAnexo(tramite, protocolo, nomeTramite) {

    var urlRequisicao = pageContext + '/pages/anexo/abrir';
    var url = pageContext + '/pages/anexo/tramite/' + tramite + '/reclamacao/' + protocolo + '.json';
    var urlDownload = pageContext + '/pages/anexo/download/tramite/' + tramite + '/reclamacao/' + protocolo + '/anexo/';

    $.get(urlRequisicao, function(data, textStatus, jqXHR) {
        data = data.replace("<meta name=\"decorator\" content=\"application_modal\" />", "");
        $("#dlgDetalheAnexo").html(data);
        $("#dlgDetalheAnexo").modal('show');

        $("#txTramite").text(nomeTramite);

        insereDatatables($("#tblAnexo"), ['nome', 'data', 'autor'], url, urlDownload, null, null, true);
    });

    return false;
}

/**
 * Detecta se houve alguma alteracao no form, armazenando este estado num campo hidden adicionado automaticamente.
 */
function detectarAlteracaoForm(form) {
    $(form + " :input").change(function() {
        marcarAlteracaoForm(form);
    });
}

function marcarAlteracaoForm(form) {
    var nomeHidden = 'houveAlteracao_' + form;
    if ($('input[name="' + nomeHidden + '"]').length == 0) {
        $(form).append('<input type="hidden" name="' + nomeHidden + '" value="true" />');
    }
}

function houverAlteracaoForm(form) {
    var nomeHidden = 'houveAlteracao_' + form;
    if ($('input[name="' + nomeHidden + '"]').length > 0 && $('input[name="' + nomeHidden + '"]').val() == 'true') {
        return true;
    }

    return false;
}

function validarTelefone(telefone) {
    if (telefone) {
        if (telefone.length < 14) {
            return false;
        }
        return true;
    }
    return false;
}

function validarEmail(email) {
    if (email) {
        var filtro = /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
        var resposta = filtro.test(email);
        if (!resposta) {
            return false;
        }
        return true;
    }
    return false;
}


function atualizarHoraInicio(idCampo) {
    $("#" + idCampo).val("00:00:00");
}

function atualizarHoraFim(idCampo) {
    $("#" + idCampo).val("23:59:59");
}

function atualizarDataInicio(idCampo) {
    var hoje = new Date();
    var mesAnterior = subtrairDias(hoje, 30);

    var dia = preencheZeros(mesAnterior.getDate() + "", 2);
    var mes = preencheZeros((mesAnterior.getMonth() + 1) + "", 2);
    var ano = preencheZeros(mesAnterior.getFullYear() + "", 4);

    var dataFim = dia + "/" + mes + "/" + ano;
    $("#" + idCampo).val(dataFim);
}

function atualizarDataFim(idCampo) {
    var hoje = new Date();
    var dia = preencheZeros(hoje.getDate() + "", 2);
    var mes = preencheZeros((hoje.getMonth() + 1) + "", 2);
    var ano = preencheZeros(hoje.getFullYear() + "", 4);

    var dataIni = dia + "/" + mes + "/" + ano;
    $("#" + idCampo).val(dataIni);
}


function validarHora(campo) {
    var resposta = false;
    var valor = campo.value;
    if (valor != null && valor != "") {
        if (valor.length < 8) {
            resposta = true;
        } else {
            var hora = valor.substring(0, 2);
            var minuto = valor.substring(3, 5);
            var segundo = valor.substring(6, 8);

            if (hora > 23) {
                resposta = true;
            } else if (minuto > 59) {
                resposta = true;
            } else if (segundo > 59) {
                resposta = true;
            }
        }
    }
    return resposta;
}

function validarCampoData(obj, inicial) {
    var data = obj.value;
    if (data.length > 0) {
        var dia = data.substring(0, 2);
        var mes = data.substring(3, 5);
        var ano = data.substring(6, 10);

        // Criando um objeto Date usando os valores ano, mes e dia.
        var novaData = new Date(ano, (mes - 1), dia);

        var mesmoDia = parseInt(dia, 10) == parseInt(novaData.getDate());
        var mesmoMes = parseInt(mes, 10) == parseInt(novaData.getMonth()) + 1;
        var mesmoAno = parseInt(ano) == parseInt(novaData.getFullYear());

        if (!((mesmoDia) && (mesmoMes) && (mesmoAno))) {
            alert('Data informada inv\u00E1lida!');

            if (inicial == null) {
                obj.value = "";
            }
            else if (inicial) {
                atualizarDataInicio();
            } else {
                atualizarDataFim();
            }

            obj.focus();
            return false;
        }
    }
    return true;
}

function exibeBotoesDownload(exibir, textoPesquisa) {

    if (exibir) {
        $("#btn-export-pdf").css('opacity', '100');
        $("#btn-export-pdf").css('cursor', 'pointer');
        $("#btn-export-pdf").attr('onclick', 'bt_export_pdf()');

        $("#btn-export-csv").css('opacity', '100');
        $("#btn-export-csv").css('cursor', 'pointer');
        $("#btn-export-csv").attr('onclick', 'bt_export_csv()');


        $("#btn-export-pdf").attr("title", "Imprimir em PDF");
        $("#btn-export-csv").attr("title", "Exportar em CSV");

        var tagEdicaoTexto = $('.tag_reclamacao');
        if (tagEdicaoTexto.attr("href") != null) {
            tagEdicaoTexto.editable();
        }
    }
    else {
        $("#btn-export-pdf").css('opacity', '.65');
        $("#btn-export-pdf").css('cursor', 'auto');
        $("#btn-export-pdf").removeAttr('onclick');

        $("#btn-export-csv").css('opacity', '.65');
        $("#btn-export-csv").css('cursor', 'auto');
        $("#btn-export-csv").removeAttr('onclick');

        if (textoPesquisa) {
            $("#btn-export-pdf").attr("title", "Campos de pesquisa alterados. Efetue a consulta para habilitar a impressao do PDF.");
            $("#btn-export-csv").attr("title", "Campos de pesquisa alterados. Efetue a consulta para habilitar a exportacao do CSV.");
        } else {
            $("#btn-export-pdf").attr("title", "Sem resultado. Refine sua busca.");
            $("#btn-export-csv").attr("title", "Sem resultado. Refine sua busca.");
        }
    }
}

function fixLabelsInteirosChartJS(data) {
    var maxValue = false;
    for (datasetIndex = 0; datasetIndex < data.datasets.length; ++datasetIndex) {
        var setMax = Math.max.apply(null, data.datasets[datasetIndex].data);
        if (maxValue === false || setMax > maxValue)
            maxValue = setMax;
    }

    var steps = new Number(maxValue);
    var stepWidth = new Number(1);
    if (maxValue > 10) {
        stepWidth = Math.floor(maxValue / 10);
        steps = Math.ceil(maxValue / stepWidth);
    }
    return {scaleOverride: true, scaleSteps: steps, scaleStepWidth: stepWidth, scaleStartValue: 0};
}

function subtrairDias(data, dias) {
    return new Date(data.getTime() - (dias * 24 * 60 * 60 * 1000));
}

function getExtraParamsForm(form, aoData) {
    $(form + " input, " + form + " select, " + form + " textarea").each(function() {
        aoData.push({"name": $(this).attr("name"), "value": $(this).val()});
    });
}

function preencheZeros(param, tamanho) {

    while (param.length < tamanho) {
        param = "0" + param;
    }
    return param;
}

function gerarCookie(strCookie, strValor) {
    $.cookie(strCookie, strValor);
}

function scrollToAnchor(el) {
    $('html,body').animate({scrollTop: $(el).offset().top});
}

/**
 * Funcao de suporte para casos de sucesso ou validacao do Ajax nos modais de pesquisas datatables. As validacoes
 * sao exibidos em caixa de mensagem inline do componente 'pwe:mensagem'.
 * 
 * @param idTabela: id da datatable.
 * @param data: dados retornados pelo ajax no caso de sucesso. Ele deve ser um JSON da classe 'PesquisaDatatableTO'.
 * @param jqXHR: dados da requisicao ajax.
 * @param fnCallback: funcao da datatable para adicionar os dados na tabela.
 * @param funcaoAdicionalSucesso: nao obrigatorio, pode-se passar uma funcao a ser executada no caso de sucesso.
 * @param funcaoAdicionalValidacao: nao obrigatorio, pode-se passar uma funcao a ser executada no caso de validacao.
 */
function pesquisaComSucessoOuValidacaoDatatable(idTabela, data, jqXHR, fnCallback, funcaoAdicionalSucesso, funcaoAdicionalValidacao) {
    exibeBotoesDownload(false);
    $("#" + idTabela + "_processing").hide();
    var dados = data['dados'];
    var validacoes = data['validacoes'];

    if (jqXHR.status == 200) {
        //sucesso, repassa os dados para o datatable.
        fnCallback(dados);

        if (dados.aaData.length > 0) {
            exibeBotoesDownload(true);
            limparMensagensTela();
        }
        else {
            exibirMensagemAlerta("N\u00E3o foram encontrados resultados.");
        }

        //fecha o modal.
        $('[data-toggle="dropdown"]').parent().removeClass('open');
        funcaoAdicionalSucesso;

    } else {
        //exibe erros abaixo dos campos com o modal ainda aberto.
        for (var campo in validacoes) {
            var mensagem = validacoes[campo];

            var componente = $("#" + campo);
            var componenteErro = $("#" + campo + "_msg_erros");
            if (componenteErro.size() < 1) {
                var spanErro = $("<span id='" + campo + ".errors' class='has-error'>" + mensagem + "</span>");
                spanErro.insertAfter(componente);
            } else {
                componenteErro.text(mensagem);
            }
        }
        funcaoAdicionalValidacao;
    }
}

function pesquisaNaoRealizadaDatatable(idTabela) {
    $('[data-toggle="dropdown"]').parent().removeClass('open');
    exibeBotoesDownload(false);
    $("#" + idTabela + "_processing").hide();
}

/**
 * Funcao de suporte para casos de erros genericos e impeditivos do Ajax nos modais de pesquisas datatables.
 * O erro nao eh exibido em alerta ou caixa de mensagem inline. Eh adicionado log no console. Isso foi feito
 * porque existem casos de perda de sessao ou erros nao necessariamente da aplicacao.
 * 
 * @param idTabela: id da datatable.
 * @param jqXHR: dados da requisicao ajax.
 * @param funcaoAdicionalErro: nao obrigatorio, pode-se passar uma funcao a ser executada no caso de erros.
 */
function pesquisaComErrosDatatable(idTabela, jqXHR, funcaoAdicionalErro) {
    console.log(jqXHR.status + ": " + jqXHR.responseText); //isto esta correto!
    var validouRedirectLogin = validarRedirectLoginErroRequisicaoPesquisas(jqXHR);
    if(!validouRedirectLogin){
        //erro generico, exibido na caixa acima da tabela.
        //fecha o modal.
        $('[data-toggle="dropdown"]').parent().removeClass('open');

        exibeBotoesDownload(false);
        $("#" + idTabela + "_processing").hide();
        funcaoAdicionalErro;
    }
}
/**
 * Atualiza uma combo box a partir de uma url com retorno json 
 * @param url - recurso a ser acessado
 * @param idCampo - campo a ser atualizado
 * @param value - nao obrigatorio, o campo ja vai vir selecionado caso o valor seja passado
 */
function atualizarComboBox(url, idCampo, value) {

    var campo = $('' + idCampo);
    $.getJSON(url).done(function(data) {

        $.each(data, function(i, item) {
            campo.append('<option value="' + item.value + '">' + item.label + '</option>');
        });
        campo.val(value);
    });
}

function ajustarTabela(id) {
    $(window).resize(function() {
        $("#" + id).dataTable().css({width: $("#" + id).dataTable().parent().width()});
        $("#" + id).dataTable().fnAdjustColumnSizing();
    });
}

var tagsToReplace = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;'
};

function replaceTag(tag) {
    return tagsToReplace[tag] || tag;
}

function escapeHTML(str) {
    return str.replace(/[&<>]/g, replaceTag);
}

function inserirValorNaPosicaoCursor(selectorComponente, texto) {
    $componenteTexto = $(selectorComponente);
    var cursorPos = $componenteTexto.prop('selectionStart');
    var v = $componenteTexto.val();
    var textBefore = v.substring(0, cursorPos);
    var textAfter = v.substring(cursorPos, v.length);
    $componenteTexto.val(textBefore + texto + textAfter);
}

function requisicaoAjaxDatatable(sSource, aoData, fnCallback, oSettings) {
    oSettings.jqXHR = $.ajax({
        "dataType": 'json',
        "type": "GET",
        "url": sSource,
        "data": aoData,
        "success": function(data, textStatus, jqXHR) {
            fnCallback(data);
        },
        "error": function(xhr, textStatus, errorThrown) {
            exibirMensagemAlertaOuAjax(xhr.status, xhr.responseText);
        }
    });
}

function requisicaoAjaxDatatableDandelion(sSource, aoData, fnCallback, oSettings) {
    oSettings.jqXHR = $.ajax({
        "dataType": 'json',
        "type": "GET",
        "url": sSource,
        "data": aoData,
        "success": function(data, textStatus, jqXHR) {
            fnCallback(data['dados']);
        },
        "error": function(xhr, textStatus, errorThrown) {
            exibirMensagemAlertaOuAjax(xhr.status, xhr.responseText);
        }
    });
}

String.prototype.replaceAll = function(search, replace) {
    //if replace is null, return original string otherwise it will replace search string with 'undefined'.
    if (!replace)
        return this;
    return this.replace(new RegExp('[' + search + ']', 'g'), replace);
};

function configurarPadroesMultiselect(){
	$('.multiselect').multiselect({
		includeSelectAllOption: true,
		numberDisplayed: 3,
		selectAllText: 'Selecionar Todos',
		nSelectedText: 'selecionados',
		nonSelectedText: 'Selecione uma ou mais op\u00E7\u00F5es...',
		allSelectedText: 'Todos selecionados.'
	});
}

/**
 * Configura dois datepicker para os dados de um refletir no outro.
 * 
 * @param string idPicker1 id do primeiro datepicker, com '#'.
 * @param string idPicker2 id do primeiro datepicker, com '#'.
 * @param string qtdDiasMinPicker1 opcional, indica ao primeiro picker uma data minima, no padrao de calculo do datapicker. Por exemplo para limite de 90 dias, '-90D'.
 * @param int qtdDiasMaxPicker2 opcional, indicando o limite de dias entre a selecao do primeiro e o maximo do segundo. Por exemplo para que apos selecionar o primeiro picker, limitar ate 31 dias do segundo, passar 31 neste parametro.
 */
function configurarPeriodosDatepicker(idPicker1, idPicker2, qtdDiasMinPicker1, qtdDiasMaxPicker2) {
    $(idPicker1).datepicker({
        minDate: qtdDiasMinPicker1
    });
    $(idPicker2).datepicker();

    $(idPicker1).change(function() {
        var dateValue = this.value;
        if ((dateValue && dateValue != null && dateValue != '') && (qtdDiasMaxPicker2 && qtdDiasMaxPicker2 != null && qtdDiasMaxPicker2 != '')) {
            var data = $.datepicker.parseDate('dd/mm/yy', dateValue);
            var contDate = new Date((data.getTime()) + (qtdDiasMaxPicker2 * 86400000));
            var maxDate = $.datepicker.formatDate("dd/mm/yy", contDate);
            $(idPicker2).datepicker("option", "maxDate", maxDate);
        }
        $(idPicker2).datepicker("option", "minDate", dateValue);
    });
    $(idPicker1).change();
}

/**
 * Limita o valor possivel de selecao anual, de 'x' atÃ© hoje. Por exemplo para
 * limitar de 120 anos atras atÃ© hoje, passar 120 no parametro.
 * 
 * @param string idPicker id do datepicker, com '#'.
 * @param int menosAnos ano de calculo para limite no passado.
 */
function limitarAnoPassadoPossivelDatepicker(idPicker, menosAnos) {
    $(idPicker).datepicker({
        changeMonth: true,
        changeYear: true,
        yearRange: (new Date().getFullYear() - menosAnos) + ':' + new Date().getFullYear()
    });
}

function validarRedirectLoginErroRequisicaoPesquisas(jqXHR){
    var urlLogin = jqXHR.getResponseHeader("url-login-redirect"); //variavel colocada pelo servidor
    if(urlLogin && urlLogin !== undefined && (urlLogin.indexOf('login') > -1 || urlLogin.indexOf('logout') > -1 || urlLogin.indexOf('sessao') > -1)){
        console.log('Redirecionando para... ', urlLogin)
        window.location.href = (pageContext+urlLogin);
        return true;
    }
    return false;
}