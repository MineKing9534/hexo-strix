"use strict";(()=>{var U=globalThis,O=U.ShadowRoot&&(U.ShadyCSS===void 0||U.ShadyCSS.nativeShadow)&&"adoptedStyleSheets"in Document.prototype&&"replace"in CSSStyleSheet.prototype,Y=Symbol(),Q=new WeakMap,R=class{constructor(e,t,s){if(this._$cssResult$=!0,s!==Y)throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");this.cssText=e,this.t=t}get styleSheet(){let e=this.o,t=this.t;if(O&&e===void 0){let s=t!==void 0&&t.length===1;s&&(e=Q.get(t)),e===void 0&&((this.o=e=new CSSStyleSheet).replaceSync(this.cssText),s&&Q.set(t,e))}return e}toString(){return this.cssText}},ee=i=>new R(typeof i=="string"?i:i+"",void 0,Y);var te=(i,e)=>{if(O)i.adoptedStyleSheets=e.map(t=>t instanceof CSSStyleSheet?t:t.styleSheet);else for(let t of e){let s=document.createElement("style"),o=U.litNonce;o!==void 0&&s.setAttribute("nonce",o),s.textContent=t.cssText,i.appendChild(s)}},z=O?i=>i:i=>i instanceof CSSStyleSheet?(e=>{let t="";for(let s of e.cssRules)t+=s.cssText;return ee(t)})(i):i;var{is:fe,defineProperty:be,getOwnPropertyDescriptor:ge,getOwnPropertyNames:me,getOwnPropertySymbols:$e,getPrototypeOf:Ae}=Object,N=globalThis,se=N.trustedTypes,_e=se?se.emptyScript:"",we=N.reactiveElementPolyfillSupport,x=(i,e)=>i,B={toAttribute(i,e){switch(e){case Boolean:i=i?_e:null;break;case Object:case Array:i=i==null?i:JSON.stringify(i)}return i},fromAttribute(i,e){let t=i;switch(e){case Boolean:t=i!==null;break;case Number:t=i===null?null:Number(i);break;case Object:case Array:try{t=JSON.parse(i)}catch{t=null}}return t}},oe=(i,e)=>!fe(i,e),ie={attribute:!0,type:String,converter:B,reflect:!1,useDefault:!1,hasChanged:oe};Symbol.metadata??=Symbol("metadata"),N.litPropertyMetadata??=new WeakMap;var f=class extends HTMLElement{static addInitializer(e){this._$Ei(),(this.l??=[]).push(e)}static get observedAttributes(){return this.finalize(),this._$Eh&&[...this._$Eh.keys()]}static createProperty(e,t=ie){if(t.state&&(t.attribute=!1),this._$Ei(),this.prototype.hasOwnProperty(e)&&((t=Object.create(t)).wrapped=!0),this.elementProperties.set(e,t),!t.noAccessor){let s=Symbol(),o=this.getPropertyDescriptor(e,s,t);o!==void 0&&be(this.prototype,e,o)}}static getPropertyDescriptor(e,t,s){let{get:o,set:a}=ge(this.prototype,e)??{get(){return this[t]},set(r){this[t]=r}};return{get:o,set(r){let c=o?.call(this);a?.call(this,r),this.requestUpdate(e,c,s)},configurable:!0,enumerable:!0}}static getPropertyOptions(e){return this.elementProperties.get(e)??ie}static _$Ei(){if(this.hasOwnProperty(x("elementProperties")))return;let e=Ae(this);e.finalize(),e.l!==void 0&&(this.l=[...e.l]),this.elementProperties=new Map(e.elementProperties)}static finalize(){if(this.hasOwnProperty(x("finalized")))return;if(this.finalized=!0,this._$Ei(),this.hasOwnProperty(x("properties"))){let t=this.properties,s=[...me(t),...$e(t)];for(let o of s)this.createProperty(o,t[o])}let e=this[Symbol.metadata];if(e!==null){let t=litPropertyMetadata.get(e);if(t!==void 0)for(let[s,o]of t)this.elementProperties.set(s,o)}this._$Eh=new Map;for(let[t,s]of this.elementProperties){let o=this._$Eu(t,s);o!==void 0&&this._$Eh.set(o,t)}this.elementStyles=this.finalizeStyles(this.styles)}static finalizeStyles(e){let t=[];if(Array.isArray(e)){let s=new Set(e.flat(1/0).reverse());for(let o of s)t.unshift(z(o))}else e!==void 0&&t.push(z(e));return t}static _$Eu(e,t){let s=t.attribute;return s===!1?void 0:typeof s=="string"?s:typeof e=="string"?e.toLowerCase():void 0}constructor(){super(),this._$Ep=void 0,this.isUpdatePending=!1,this.hasUpdated=!1,this._$Em=null,this._$Ev()}_$Ev(){this._$ES=new Promise(e=>this.enableUpdating=e),this._$AL=new Map,this._$E_(),this.requestUpdate(),this.constructor.l?.forEach(e=>e(this))}addController(e){(this._$EO??=new Set).add(e),this.renderRoot!==void 0&&this.isConnected&&e.hostConnected?.()}removeController(e){this._$EO?.delete(e)}_$E_(){let e=new Map,t=this.constructor.elementProperties;for(let s of t.keys())this.hasOwnProperty(s)&&(e.set(s,this[s]),delete this[s]);e.size>0&&(this._$Ep=e)}createRenderRoot(){let e=this.shadowRoot??this.attachShadow(this.constructor.shadowRootOptions);return te(e,this.constructor.elementStyles),e}connectedCallback(){this.renderRoot??=this.createRenderRoot(),this.enableUpdating(!0),this._$EO?.forEach(e=>e.hostConnected?.())}enableUpdating(e){}disconnectedCallback(){this._$EO?.forEach(e=>e.hostDisconnected?.())}attributeChangedCallback(e,t,s){this._$AK(e,s)}_$ET(e,t){let s=this.constructor.elementProperties.get(e),o=this.constructor._$Eu(e,s);if(o!==void 0&&s.reflect===!0){let a=(s.converter?.toAttribute!==void 0?s.converter:B).toAttribute(t,s.type);this._$Em=e,a==null?this.removeAttribute(o):this.setAttribute(o,a),this._$Em=null}}_$AK(e,t){let s=this.constructor,o=s._$Eh.get(e);if(o!==void 0&&this._$Em!==o){let a=s.getPropertyOptions(o),r=typeof a.converter=="function"?{fromAttribute:a.converter}:a.converter?.fromAttribute!==void 0?a.converter:B;this._$Em=o;let c=r.fromAttribute(t,a.type);this[o]=c??this._$Ej?.get(o)??c,this._$Em=null}}requestUpdate(e,t,s,o=!1,a){if(e!==void 0){let r=this.constructor;if(o===!1&&(a=this[e]),s??=r.getPropertyOptions(e),!((s.hasChanged??oe)(a,t)||s.useDefault&&s.reflect&&a===this._$Ej?.get(e)&&!this.hasAttribute(r._$Eu(e,s))))return;this.C(e,t,s)}this.isUpdatePending===!1&&(this._$ES=this._$EP())}C(e,t,{useDefault:s,reflect:o,wrapped:a},r){s&&!(this._$Ej??=new Map).has(e)&&(this._$Ej.set(e,r??t??this[e]),a!==!0||r!==void 0)||(this._$AL.has(e)||(this.hasUpdated||s||(t=void 0),this._$AL.set(e,t)),o===!0&&this._$Em!==e&&(this._$Eq??=new Set).add(e))}async _$EP(){this.isUpdatePending=!0;try{await this._$ES}catch(t){Promise.reject(t)}let e=this.scheduleUpdate();return e!=null&&await e,!this.isUpdatePending}scheduleUpdate(){return this.performUpdate()}performUpdate(){if(!this.isUpdatePending)return;if(!this.hasUpdated){if(this.renderRoot??=this.createRenderRoot(),this._$Ep){for(let[o,a]of this._$Ep)this[o]=a;this._$Ep=void 0}let s=this.constructor.elementProperties;if(s.size>0)for(let[o,a]of s){let{wrapped:r}=a,c=this[o];r!==!0||this._$AL.has(o)||c===void 0||this.C(o,void 0,a,c)}}let e=!1,t=this._$AL;try{e=this.shouldUpdate(t),e?(this.willUpdate(t),this._$EO?.forEach(s=>s.hostUpdate?.()),this.update(t)):this._$EM()}catch(s){throw e=!1,this._$EM(),s}e&&this._$AE(t)}willUpdate(e){}_$AE(e){this._$EO?.forEach(t=>t.hostUpdated?.()),this.hasUpdated||(this.hasUpdated=!0,this.firstUpdated(e)),this.updated(e)}_$EM(){this._$AL=new Map,this.isUpdatePending=!1}get updateComplete(){return this.getUpdateComplete()}getUpdateComplete(){return this._$ES}shouldUpdate(e){return!0}update(e){this._$Eq&&=this._$Eq.forEach(t=>this._$ET(t,this[t])),this._$EM()}updated(e){}firstUpdated(e){}};f.elementStyles=[],f.shadowRootOptions={mode:"open"},f[x("elementProperties")]=new Map,f[x("finalized")]=new Map,we?.({ReactiveElement:f}),(N.reactiveElementVersions??=[]).push("2.1.2");var G=globalThis,ae=i=>i,D=G.trustedTypes,ne=D?D.createPolicy("lit-html",{createHTML:i=>i}):void 0,he="$lit$",g=`lit$${Math.random().toFixed(9).slice(2)}$`,ue="?"+g,Ee=`<${ue}>`,A=document,k=()=>A.createComment(""),C=i=>i===null||typeof i!="object"&&typeof i!="function",q=Array.isArray,Se=i=>q(i)||typeof i?.[Symbol.iterator]=="function",I=`[ 	
\f\r]`,P=/<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g,re=/-->/g,le=/>/g,m=RegExp(`>|${I}(?:([^\\s"'>=/]+)(${I}*=${I}*(?:[^ 	
\f\r"'\`<>=]|("|')|))|$)`,"g"),de=/'/g,ce=/"/g,ve=/^(?:script|style|textarea|title)$/i,Z=i=>(e,...t)=>({_$litType$:i,strings:e,values:t}),v=Z(1),Ve=Z(2),Ke=Z(3),_=Symbol.for("lit-noChange"),h=Symbol.for("lit-nothing"),pe=new WeakMap,$=A.createTreeWalker(A,129);function ye(i,e){if(!q(i)||!i.hasOwnProperty("raw"))throw Error("invalid template strings array");return ne!==void 0?ne.createHTML(e):e}var xe=(i,e)=>{let t=i.length-1,s=[],o,a=e===2?"<svg>":e===3?"<math>":"",r=P;for(let c=0;c<t;c++){let l=i[c],p,u,d=-1,y=0;for(;y<l.length&&(r.lastIndex=y,u=r.exec(l),u!==null);)y=r.lastIndex,r===P?u[1]==="!--"?r=re:u[1]!==void 0?r=le:u[2]!==void 0?(ve.test(u[2])&&(o=RegExp("</"+u[2],"g")),r=m):u[3]!==void 0&&(r=m):r===m?u[0]===">"?(r=o??P,d=-1):u[1]===void 0?d=-2:(d=r.lastIndex-u[2].length,p=u[1],r=u[3]===void 0?m:u[3]==='"'?ce:de):r===ce||r===de?r=m:r===re||r===le?r=P:(r=m,o=void 0);let b=r===m&&i[c+1].startsWith("/>")?" ":"";a+=r===P?l+Ee:d>=0?(s.push(p),l.slice(0,d)+he+l.slice(d)+g+b):l+g+(d===-2?c:b)}return[ye(i,a+(i[t]||"<?>")+(e===2?"</svg>":e===3?"</math>":"")),s]},T=class i{constructor({strings:e,_$litType$:t},s){let o;this.parts=[];let a=0,r=0,c=e.length-1,l=this.parts,[p,u]=xe(e,t);if(this.el=i.createElement(p,s),$.currentNode=this.el.content,t===2||t===3){let d=this.el.content.firstChild;d.replaceWith(...d.childNodes)}for(;(o=$.nextNode())!==null&&l.length<c;){if(o.nodeType===1){if(o.hasAttributes())for(let d of o.getAttributeNames())if(d.endsWith(he)){let y=u[r++],b=o.getAttribute(d).split(g),M=/([.?@])?(.*)/.exec(y);l.push({type:1,index:a,name:M[2],strings:b,ctor:M[1]==="."?j:M[1]==="?"?V:M[1]==="@"?K:E}),o.removeAttribute(d)}else d.startsWith(g)&&(l.push({type:6,index:a}),o.removeAttribute(d));if(ve.test(o.tagName)){let d=o.textContent.split(g),y=d.length-1;if(y>0){o.textContent=D?D.emptyScript:"";for(let b=0;b<y;b++)o.append(d[b],k()),$.nextNode(),l.push({type:2,index:++a});o.append(d[y],k())}}}else if(o.nodeType===8)if(o.data===ue)l.push({type:2,index:a});else{let d=-1;for(;(d=o.data.indexOf(g,d+1))!==-1;)l.push({type:7,index:a}),d+=g.length-1}a++}}static createElement(e,t){let s=A.createElement("template");return s.innerHTML=e,s}};function w(i,e,t=i,s){if(e===_)return e;let o=s!==void 0?t._$Co?.[s]:t._$Cl,a=C(e)?void 0:e._$litDirective$;return o?.constructor!==a&&(o?._$AO?.(!1),a===void 0?o=void 0:(o=new a(i),o._$AT(i,t,s)),s!==void 0?(t._$Co??=[])[s]=o:t._$Cl=o),o!==void 0&&(e=w(i,o._$AS(i,e.values),o,s)),e}var F=class{constructor(e,t){this._$AV=[],this._$AN=void 0,this._$AD=e,this._$AM=t}get parentNode(){return this._$AM.parentNode}get _$AU(){return this._$AM._$AU}u(e){let{el:{content:t},parts:s}=this._$AD,o=(e?.creationScope??A).importNode(t,!0);$.currentNode=o;let a=$.nextNode(),r=0,c=0,l=s[0];for(;l!==void 0;){if(r===l.index){let p;l.type===2?p=new H(a,a.nextSibling,this,e):l.type===1?p=new l.ctor(a,l.name,l.strings,this,e):l.type===6&&(p=new W(a,this,e)),this._$AV.push(p),l=s[++c]}r!==l?.index&&(a=$.nextNode(),r++)}return $.currentNode=A,o}p(e){let t=0;for(let s of this._$AV)s!==void 0&&(s.strings!==void 0?(s._$AI(e,s,t),t+=s.strings.length-2):s._$AI(e[t])),t++}},H=class i{get _$AU(){return this._$AM?._$AU??this._$Cv}constructor(e,t,s,o){this.type=2,this._$AH=h,this._$AN=void 0,this._$AA=e,this._$AB=t,this._$AM=s,this.options=o,this._$Cv=o?.isConnected??!0}get parentNode(){let e=this._$AA.parentNode,t=this._$AM;return t!==void 0&&e?.nodeType===11&&(e=t.parentNode),e}get startNode(){return this._$AA}get endNode(){return this._$AB}_$AI(e,t=this){e=w(this,e,t),C(e)?e===h||e==null||e===""?(this._$AH!==h&&this._$AR(),this._$AH=h):e!==this._$AH&&e!==_&&this._(e):e._$litType$!==void 0?this.$(e):e.nodeType!==void 0?this.T(e):Se(e)?this.k(e):this._(e)}O(e){return this._$AA.parentNode.insertBefore(e,this._$AB)}T(e){this._$AH!==e&&(this._$AR(),this._$AH=this.O(e))}_(e){this._$AH!==h&&C(this._$AH)?this._$AA.nextSibling.data=e:this.T(A.createTextNode(e)),this._$AH=e}$(e){let{values:t,_$litType$:s}=e,o=typeof s=="number"?this._$AC(e):(s.el===void 0&&(s.el=T.createElement(ye(s.h,s.h[0]),this.options)),s);if(this._$AH?._$AD===o)this._$AH.p(t);else{let a=new F(o,this),r=a.u(this.options);a.p(t),this.T(r),this._$AH=a}}_$AC(e){let t=pe.get(e.strings);return t===void 0&&pe.set(e.strings,t=new T(e)),t}k(e){q(this._$AH)||(this._$AH=[],this._$AR());let t=this._$AH,s,o=0;for(let a of e)o===t.length?t.push(s=new i(this.O(k()),this.O(k()),this,this.options)):s=t[o],s._$AI(a),o++;o<t.length&&(this._$AR(s&&s._$AB.nextSibling,o),t.length=o)}_$AR(e=this._$AA.nextSibling,t){for(this._$AP?.(!1,!0,t);e!==this._$AB;){let s=ae(e).nextSibling;ae(e).remove(),e=s}}setConnected(e){this._$AM===void 0&&(this._$Cv=e,this._$AP?.(e))}},E=class{get tagName(){return this.element.tagName}get _$AU(){return this._$AM._$AU}constructor(e,t,s,o,a){this.type=1,this._$AH=h,this._$AN=void 0,this.element=e,this.name=t,this._$AM=o,this.options=a,s.length>2||s[0]!==""||s[1]!==""?(this._$AH=Array(s.length-1).fill(new String),this.strings=s):this._$AH=h}_$AI(e,t=this,s,o){let a=this.strings,r=!1;if(a===void 0)e=w(this,e,t,0),r=!C(e)||e!==this._$AH&&e!==_,r&&(this._$AH=e);else{let c=e,l,p;for(e=a[0],l=0;l<a.length-1;l++)p=w(this,c[s+l],t,l),p===_&&(p=this._$AH[l]),r||=!C(p)||p!==this._$AH[l],p===h?e=h:e!==h&&(e+=(p??"")+a[l+1]),this._$AH[l]=p}r&&!o&&this.j(e)}j(e){e===h?this.element.removeAttribute(this.name):this.element.setAttribute(this.name,e??"")}},j=class extends E{constructor(){super(...arguments),this.type=3}j(e){this.element[this.name]=e===h?void 0:e}},V=class extends E{constructor(){super(...arguments),this.type=4}j(e){this.element.toggleAttribute(this.name,!!e&&e!==h)}},K=class extends E{constructor(e,t,s,o,a){super(e,t,s,o,a),this.type=5}_$AI(e,t=this){if((e=w(this,e,t,0)??h)===_)return;let s=this._$AH,o=e===h&&s!==h||e.capture!==s.capture||e.once!==s.once||e.passive!==s.passive,a=e!==h&&(s===h||o);o&&this.element.removeEventListener(this.name,this,s),a&&this.element.addEventListener(this.name,this,e),this._$AH=e}handleEvent(e){typeof this._$AH=="function"?this._$AH.call(this.options?.host??this.element,e):this._$AH.handleEvent(e)}},W=class{constructor(e,t,s){this.element=e,this.type=6,this._$AN=void 0,this._$AM=t,this.options=s}get _$AU(){return this._$AM._$AU}_$AI(e){w(this,e)}};var Pe=G.litHtmlPolyfillSupport;Pe?.(T,H),(G.litHtmlVersions??=[]).push("3.3.3");var L=(i,e,t)=>{let s=t?.renderBefore??e,o=s._$litPart$;if(o===void 0){let a=t?.renderBefore??null;s._$litPart$=o=new H(e.insertBefore(k(),a),a,void 0,t??{})}return o._$AI(i),o};var X=globalThis,S=class extends f{constructor(){super(...arguments),this.renderOptions={host:this},this._$Do=void 0}createRenderRoot(){let e=super.createRenderRoot();return this.renderOptions.renderBefore??=e.firstChild,e}update(e){let t=this.render();this.hasUpdated||(this.renderOptions.isConnected=this.isConnected),super.update(e),this._$Do=L(t,this.renderRoot,this.renderOptions)}connectedCallback(){super.connectedCallback(),this._$Do?.setConnected(!0)}disconnectedCallback(){super.disconnectedCallback(),this._$Do?.setConnected(!1)}render(){return _}};S._$litElement$=!0,S.finalized=!0,X.litElementHydrateSupport?.({LitElement:S});var ke=X.litElementPolyfillSupport;ke?.({LitElement:S});(X.litElementVersions??=[]).push("4.2.2");function n(i,...e){let t=window[i];if(typeof t!="function")throw new Error(`UI action ${String(i)} is not available`);t(...e)}function Ce(i){i.key!=="Enter"&&i.key!==" "||(i.preventDefault(),n("toggleAnalysisSheet",i))}var Te=()=>v`
  <div id="topbar">
    <h1><span class="mark">He<span class="mark-x">X</span>O</span><span class="mark-tag">Observatory</span></h1>
    <span id="view-label">Analysis</span>
    <span id="status">Loading…</span>
    <span id="difficulty-badge" hidden></span>
    <button id="resign-btn" hidden @click=${()=>n("resign")}>Resign</button>
    <button id="copy-htttx-btn" hidden @click=${()=>n("copyHtttx")}>Copy game record</button>
    <button id="analyze-game-btn" hidden @click=${()=>n("analyzeThisGame")}>Analyze this game</button>
    <button id="new-game-btn" @click=${()=>n("openModal")}>New game</button>
    <button id="analysis-btn" @click=${()=>n("goToAnalysis")}>Analysis</button>
    <button id="play-btn" @click=${()=>n("goToPlay")}>&larr; Play</button>
    <button id="topbar-menu-btn" type="button" aria-haspopup="true" aria-expanded="false"
      aria-label="Open menu" @click=${i=>n("toggleTopbarMenu",i)}>
      <svg viewBox="0 0 18 18" aria-hidden="true" focusable="false">
        <rect x="2" y="4" width="14" height="1.6" rx="0.8" fill="currentColor"></rect>
        <rect x="2" y="8.2" width="14" height="1.6" rx="0.8" fill="currentColor"></rect>
        <rect x="2" y="12.4" width="14" height="1.6" rx="0.8" fill="currentColor"></rect>
      </svg>
    </button>
    <div id="topbar-menu" role="menu" aria-label="More actions"></div>
    <a id="gh-link" href="https://github.com/SootyOwl/hexo-strix" target="_blank"
      rel="noopener noreferrer" title="View source on GitHub" aria-label="View source on GitHub">
      <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" focusable="false"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
      <span class="gh-label">Source</span>
    </a>
  </div>
`,He=()=>v`
  <div id="analysis-controls">
    <div id="analysis-sheet-handle" data-label="Controls" aria-label="Toggle controls panel"
      role="button" tabindex="0" @click=${i=>n("toggleAnalysisSheet",i)}
      @keydown=${Ce}></div>
    <div class="analysis-mode-tabs" role="tablist" aria-label="Analysis tools">
      <button id="analysis-mode-analysis" type="button" role="tab" aria-selected="true"
        aria-controls="analysis-controls-body" @click=${()=>n("closeProofLab")}>Analysis</button>
      <button id="proof-lab-launch" type="button" role="tab" aria-selected="false"
        aria-controls="proof-lab-drawer" disabled @click=${()=>n("openProofLab")}>
        <span class="proof-lab-launch-icon" aria-hidden="true">◇</span> Proof lab
      </button>
    </div>
    <div id="analysis-info"></div>
    <div id="analysis-position-browser" aria-label="Position navigation">
      <div id="analysis-eval-wrap" hidden>
        <canvas id="analysis-eval-bar" width="320" height="48" tabindex="0" role="slider"
          aria-label="Game position timeline" aria-valuemin="1" aria-valuemax="1" aria-valuenow="1"
          @pointerdown=${i=>n("onAnalysisEvalPointerDown",i)}
          @pointermove=${i=>n("onAnalysisEvalPointerMove",i)}
          @pointerup=${i=>n("onAnalysisEvalPointerUp",i)}
          @pointercancel=${i=>n("onAnalysisEvalPointerUp",i)}
          @pointerleave=${()=>n("onAnalysisEvalPointerLeave")}
          @click=${i=>n("onAnalysisEvalClick",i)}
          @keydown=${i=>n("onAnalysisEvalKeydown",i)}></canvas>
        <div id="analysis-eval-preview" hidden></div>
      </div>
      <div id="analysis-navigation" class="row" hidden>
        <button id="analysis-previous-position" class="analysis-nav-icon" type="button"
          aria-label="Previous position" title="Previous position" @click=${()=>n("analysisUndo")}>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M9 4 3 10l6 6M4 10h13"/></svg>
        </button>
        <button id="analysis-latest-mainline" class="analysis-nav-icon" type="button"
          aria-label="Latest position in the game" title="Latest position in the game"
          @click=${()=>n("returnToMainline")}>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 4 6 6-6 6M14 4v12"/></svg>
        </button>
      </div>
    </div>
    <div id="analysis-controls-body" role="tabpanel" aria-labelledby="analysis-mode-analysis">
      <div id="analysis-source-summary" hidden>
        <span><strong id="analysis-source-title">Loaded game</strong><small id="analysis-source-meta"></small></span>
        <button type="button" class="secondary-button" @click=${()=>n("editAnalysisSource")}>Change</button>
      </div>
      <div id="analysis-setup">
        <button id="hds-import-trigger" type="button" @click=${()=>n("openHdsImport")}>
          Import from Hexo sandbox <span aria-hidden="true">&rarr;</span>
        </button>
        <label class="analysis-record-field" for="analysis-htttx">
          <span class="field-label">Paste a game record (HTTTX)</span>
          <span id="analysis-record-hint" class="field-hint">Open the game now. Analysis starts only when you ask for it.</span>
          <textarea id="analysis-htttx" rows="4" placeholder="version[1];\n1. [1,0][2,0];\n..."
            aria-describedby="analysis-record-hint"
            @input=${()=>n("analysisInputChanged")}></textarea>
        </label>
        <div class="analysis-setup-actions">
          <button id="analysis-source-cancel" type="button" class="secondary-button" hidden
            @click=${()=>n("cancelAnalysisSourceEdit")}>Cancel</button>
          <button id="analysis-load-btn" class="primary-button" @click=${()=>n("loadGame")}>Load game</button>
        </div>
      </div>
      <div class="analysis-run-group">
        <div class="analysis-run-actions">
          <button id="analysis-position-btn" @click=${()=>n("analyzeCurrentPosition")} disabled>Analyze position</button>
          <button id="analysis-game-btn" @click=${()=>n("analyzeWholeGame")} disabled>Analyze full game</button>
        </div>
      </div>
      <details class="analysis-advanced analysis-settings">
        <summary>
          <span>Settings</span>
          <small id="analysis-settings-status">Standard · auto off</small>
        </summary>
        <div class="analysis-settings-body">
          <section class="analysis-settings-section" aria-labelledby="analysis-search-settings-title">
            <h3 id="analysis-search-settings-title">Analysis</h3>
            <label class="analysis-strength-field" for="analysis-strength">
              <span class="field-label">Analysis effort</span>
              <span id="analysis-strength-hint" class="field-hint">Higher settings examine more possible continuations and take longer. Instant gives an estimate without searching ahead.</span>
              <select id="analysis-strength" aria-describedby="analysis-strength-hint" @change=${()=>n("saveAnalysisStrength")}>
                <option value="network">Instant · no search</option>
                <option value="quick">Quick</option>
                <option value="standard" selected>Standard</option>
                <option value="strong">Strong</option>
                <option value="deep">Deep</option>
              </select>
            </label>
            <label class="analysis-setting-toggle" for="analysis-auto-branch">
              <input id="analysis-auto-branch" type="checkbox"
                @change=${()=>n("saveAutomaticAnalysis")}>
              <span><strong>Analyze new moves automatically</strong><small>Start analysis after you place a hex</small></span>
            </label>
            <label class="analysis-setting-toggle" for="analysis-auto-forcing">
              <input id="analysis-auto-forcing" type="checkbox" checked
                @change=${()=>n("saveAutomaticForcing")}>
              <span><strong>Check for forced wins</strong><small>Look for a win the opponent cannot stop</small></span>
            </label>
          </section>
          <section class="analysis-settings-section" aria-labelledby="analysis-display-settings-title">
            <h3 id="analysis-display-settings-title">Board overlays</h3>
            <div class="analysis-display-options-body">
              <label><input type="checkbox" id="analysis-heatmap" checked @change=${()=>n("saveDisplayPreferences")}> Suggested moves</label>
              <label><input type="checkbox" id="analysis-forcing" checked @change=${()=>n("saveDisplayPreferences")}> Winning lines</label>
              <label><input type="checkbox" id="analysis-threats" @change=${()=>n("saveDisplayPreferences")}> Threats to answer</label>
            </div>
          </section>
        </div>
      </details>
      <div id="analysis-progress">
        <div id="analysis-progress-track"><div id="analysis-progress-bar"></div></div>
        <div id="analysis-progress-label"></div>
      </div>
      <div id="analysis-movetree"></div>
      <details class="analysis-advanced analysis-reading-guide">
        <summary>How to read analysis</summary>
        <div id="analysis-caveat">The score shows who the computer expects to win: positive favours P1 and negative favours P2. Point to or choose the graph to view a position. Darker suggested moves are preferred. Choose any empty hex to try that move. At the end of a turn: ★ best, ✓ good, ? mistake, ✗ blunder.</div>
      </details>
    </div>
    ${Re()}
  </div>
`,Me=()=>v`
  <dialog id="hds-import-dialog" aria-labelledby="hds-import-title"
    @click=${i=>{i.target===i.currentTarget&&n("closeHdsImport")}}>
    <form @submit=${i=>{i.preventDefault(),n("convertHds")}}>
      <header class="hds-dialog-header">
        <div>
          <h2 id="hds-import-title">Import from Hexo sandbox</h2>
          <p>Paste the position's hexo.did.science link or short code.</p>
        </div>
        <button class="dialog-close" type="button" aria-label="Close import dialog"
          @click=${()=>n("closeHdsImport")}>Close</button>
      </header>
      <label for="hds-input"><span class="field-label">Sandbox link or code</span>
        <input id="hds-input" type="text" inputmode="url" autocomplete="off"
          placeholder="https://hexo.did.science/sandbox/5knldz6">
      </label>
      <div id="hds-status" role="status" aria-live="polite"></div>
      <footer class="hds-dialog-actions">
        <button class="secondary-button" type="button" @click=${()=>n("closeHdsImport")}>Cancel</button>
        <button class="primary-button" type="submit">Import position</button>
      </footer>
    </form>
  </dialog>
`,Ue=()=>v`
  <div id="analysis-forcing-depth-control" class="proof-lab-form">
    <section class="proof-lab-settings" aria-labelledby="proof-search-settings-title">
      <h3 id="proof-search-settings-title">Search settings</h3>
      <label for="analysis-forcing-engine">Search method
        <span id="analysis-forcing-engine-hint" class="field-hint">PDS-PN saves every checked reply, so you can explore, share, or download the result.</span>
        <select id="analysis-forcing-engine" aria-describedby="analysis-forcing-engine-hint" @change=${()=>n("updateForcingSolverUi")}>
          <option value="pdspn">Explore every reply · PDS-PN</option>
          <option value="idtt" selected>Find the shortest win · IDTT</option>
          <option value="dfpn">Find any win · DFPN</option>
          <option value="pdspn-shortest">Confirm the shortest win · PDS-PN</option>
          <option value="pns">Second yes-or-no check · PNS</option>
        </select>
      </label>
      <label for="analysis-forcing-width">Moves to consider
        <select id="analysis-forcing-width">
          <option value="wide" selected>Broad · all legal moves</option>
          <option value="tight">Direct only · immediate threats</option>
        </select>
      </label>
      <label id="analysis-forcing-depth-row" for="analysis-forcing-depth">
        <span id="analysis-forcing-depth-label">Maximum turns by the winning side</span>
        <span class="analysis-depth-input"><input id="analysis-forcing-depth" type="number" min="1" max="60" value="12" step="1" inputmode="numeric"> turns</span>
      </label>
      <label for="analysis-forcing-budget"><span id="analysis-forcing-budget-label">Search effort</span>
        <select id="analysis-forcing-budget">
          <option value="20000">Quick · 20,000 steps</option>
          <option value="250000" selected>Standard · 250,000 steps</option>
          <option value="1000000">Thorough · 1 million steps</option>
          <option value="5000000">Deep · 5 million steps</option>
          <option value="25000000">Extended · 25 million steps</option>
          <option value="100000000">Very long · 100 million steps</option>
        </select>
      </label>
      <label id="analysis-forcing-leaf-row" for="analysis-forcing-leaf-budget" hidden>
        <span>Extra search for each branch</span>
        <select id="analysis-forcing-leaf-budget">
          <option value="1000">Light · 1,000 steps</option>
          <option value="2000" selected>Balanced · 2,000 steps</option>
          <option value="5000">Thorough · 5,000 steps</option>
          <option value="10000">Deep · 10,000 steps</option>
          <option value="25000">Extended · 25,000 steps</option>
          <option value="50000">Very deep · 50,000 steps</option>
        </select>
      </label>
    </section>
    <div class="analysis-solver-actions">
      <button id="analysis-solve-forcing-btn" @click=${()=>n("solveCurrentForcing")}>Check for a forced win</button>
      <button id="analysis-cancel-forcing-btn" @click=${()=>n("cancelForcingSolve")} hidden>Stop search</button>
      <button id="analysis-explore-certificate-btn" @click=${()=>n("openProofExplorer")} hidden>View all replies</button>
      <button id="analysis-share-certificate-btn" @click=${()=>n("shareForcingCertificate")} hidden>Copy result link</button>
      <button id="analysis-download-certificate-btn" @click=${()=>n("downloadForcingCertificate")} hidden>Download result</button>
      <span id="proof-share-status" class="proof-share-status" role="status" aria-live="polite"></span>
    </div>
    <div id="analysis-forcing-status" role="status" aria-live="polite">Ready. This search runs on your device.</div>
    <details class="analysis-advanced proof-lab-help">
      <summary>How the search works</summary>
      <div class="proof-lab-help-body">
        <p id="analysis-solver-help">This method finds the shortest win, up to the maximum number of turns you set. It counts only turns taken by the side trying to win.</p>
        <p>Search effort limits the number of calculations, not the number of seconds. Broad search considers every legal move. Direct-only search considers moves that create an immediate threat. Broad search is more complete but may take longer.</p>
      </div>
    </details>
  </div>
`,Re=()=>v`
  <aside id="proof-lab-drawer" hidden role="tabpanel" aria-labelledby="proof-lab-launch">
    <header class="proof-lab-header">
      <div>
        <div class="proof-lab-title-row">
          <h2 id="proof-lab-title">Forced-win proof lab</h2>
          <span class="analysis-local-badge">on this device</span>
        </div>
        <p id="proof-lab-position">Selected analysis position</p>
        <p class="proof-lab-intro">Check whether the player to move can force a win that the opponent cannot stop.</p>
      </div>
    </header>
    ${Ue()}
  </aside>
`,Oe=()=>v`
  <div id="analysis-panel" hidden>
    ${He()}
    <div id="analysis-board-container">
      <div id="analysis-empty-state">
        <strong>Load a game to explore it</strong>
        <span>Loading is instant and does not start the analysis engine.</span>
      </div>
      <div id="gauge-wrap" hidden>
        <div class="gauge-poles">
          <span class="pole pole-p1">● P1 <b id="gauge-v1">+0.00</b></span>
          <span class="pole pole-p2"><b id="gauge-v2">−0.00</b> P2 ●</span>
        </div>
        <div class="gauge"><div class="gauge-zero"></div><div class="gauge-needle" id="gauge-needle"></div></div>
        <div class="gauge-scale"><span>P1 +1.0</span><span>EVEN</span><span>P2 +1.0</span></div>
      </div>
      <svg id="analysis-board"></svg>
      <div id="analysis-thinking" role="status" aria-live="polite" aria-atomic="true" hidden>
        <span class="analysis-thinking-mark" aria-hidden="true"><i></i><i></i><i></i></span>
        <span id="analysis-thinking-label">Checking position…</span>
      </div>
      <div id="board-legend" hidden>
        <span><i class="sw sw-p1"></i>P1 to move</span>
        <span><i class="sw sw-p2"></i>P2 to move</span>
        <span><i class="sw sw-pick"></i>top suggestion</span>
      </div>
    </div>
  </div>
`,Ne=()=>v`
  <div id="proof-explorer" role="dialog" aria-modal="true" aria-labelledby="proof-explorer-title" hidden>
    <section id="proof-board-container" aria-label="Proof position board">
      <svg id="proof-board" aria-label="HeXO proof board"></svg>
      <div class="proof-explorer-actions" aria-label="Proof explorer actions">
        <button id="proof-share-btn" @click=${()=>n("shareForcingCertificate")}
          title="Save this result and copy its link">Copy link</button>
        <button @click=${()=>n("downloadForcingCertificate")}>Download</button>
        <button id="proof-close-btn" class="proof-close" @click=${()=>n("closeProofExplorer")}
          aria-label="Close proof explorer">Close</button>
      </div>
      <div class="proof-board-tools" aria-label="Board zoom controls">
        <button @click=${()=>n("proofZoom",1.25)} aria-label="Zoom in">+</button>
        <button @click=${()=>n("proofZoom",.8)} aria-label="Zoom out">−</button>
        <button @click=${()=>n("proofFitBoard")}>Fit</button>
      </div>
      <div class="proof-board-legend">
        <span><i id="proof-attacker-swatch" class="proof-sw"></i><span id="proof-attacker-legend">winning side</span></span>
        <span><i id="proof-defender-swatch" class="proof-sw"></i><span id="proof-defender-legend">defending side</span></span>
        <span><i class="proof-sw proof-sw-choice"></i>previewed move</span>
      </div>
      <aside class="proof-explorer-panel" aria-label="Proof navigation">
        <header class="proof-explorer-heading">
          <span class="proof-explorer-kicker">Checked winning strategy</span>
          <h2 id="proof-explorer-title">Explore the win</h2>
          <span id="proof-explorer-summary"></span>
        </header>
        <nav class="proof-history-actions" aria-label="Proof history">
          <button id="proof-back-btn" @click=${()=>n("proofExplorerBack")} title="Go back one step">&larr; Back</button>
          <button @click=${()=>n("proofExplorerReset")} title="Return to the first position">Start again</button>
        </nav>
        <div class="proof-progress-copy"><span id="proof-progress-label"></span><span id="proof-node-label"></span></div>
        <div class="proof-progress-track"><div id="proof-progress-bar"></div></div>
        <div id="proof-optimization-note" class="proof-optimization-note" hidden></div>
        <div id="proof-step-card"></div>
        <div class="proof-path-heading"><span>Proof path</span><small><span class="proof-hover-hint">hover to preview · </span>choose to follow</small></div>
        <div id="proof-tree" class="proof-tree" role="tree" aria-label="Positions and available branches"></div>
        <div class="proof-panel-actions">
          <button id="proof-shortest-line-btn" @click=${()=>n("proofExplorerToggleShortestLine")} hidden>Longest defence</button>
          <button id="proof-worst-btn" class="proof-primary" @click=${()=>n("proofExplorerWorstCase")}
            title="Follow the reply that delays the win longest">Choose longest defence &rarr;</button>
        </div>
        <details class="proof-explorer-note">
          <summary>How to read this proof</summary>
          <p>On the winning side's turn, each branch shown is a move that this search proved will win. On the other side's turn, every checked reply is shown. “Longest defence” follows the reply that delays the win for the most turns.</p>
        </details>
      </aside>
    </section>
  </div>
`,De=()=>v`
  <div id="modal-bg">
    <div id="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <h2 id="modal-title">New game</h2>
      <div id="bot-stats">
        <div id="bot-stats-current">Loading the bot's record…</div>
        <div id="bot-stats-alltime"></div>
      </div>
      <label for="modal-name"><span class="field-label">Name</span><span class="field-optional">Optional</span><input id="modal-name" type="text" maxlength="64" autocomplete="off"></label>
      <label for="modal-elo"><span class="field-label">Your rating (Elo)</span><span class="field-optional">Optional · enter your own estimate</span><input id="modal-elo" type="number" min="0" max="3500" placeholder="1500" autocomplete="off" inputmode="numeric"></label>
      <fieldset class="side-fieldset">
        <legend>Side</legend>
        <div class="side-row">
          <button class="side-btn" data-side="P1" @click=${()=>n("selectSide","P1")}><span class="stone stone-p1">●</span>P1 <span class="side-colour">orange</span></button>
          <button class="side-btn selected" data-side="random" @click=${()=>n("selectSide","random")}><span class="stone">?</span>Random</button>
          <button class="side-btn" data-side="P2" @click=${()=>n("selectSide","P2")}><span class="stone stone-p2">●</span>P2 <span class="side-colour">blue</span></button>
        </div>
      </fieldset>
      <label id="diff-label" hidden>Search effort</label>
      <div id="diff-row" class="diff-row" hidden></div>
      <button id="start-btn" @click=${()=>n("startGame")}>Start game</button>
    </div>
  </div>
`,Le=()=>v`
  ${Te()}
  <div id="board-container"><svg id="board"></svg></div>
  ${Oe()}
  ${Ne()}
  ${Me()}
  ${De()}
`,J=class extends HTMLElement{connectedCallback(){L(Le(),this)}};customElements.define("hexo-observatory-app",J);})();
/*! Bundled license information:

@lit/reactive-element/css-tag.js:
  (**
   * @license
   * Copyright 2019 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/reactive-element.js:
lit-html/lit-html.js:
lit-element/lit-element.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/is-server.js:
  (**
   * @license
   * Copyright 2022 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)
*/
