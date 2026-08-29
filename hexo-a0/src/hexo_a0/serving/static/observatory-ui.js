"use strict";(()=>{var U=globalThis,N=U.ShadowRoot&&(U.ShadyCSS===void 0||U.ShadyCSS.nativeShadow)&&"adoptedStyleSheets"in Document.prototype&&"replace"in CSSStyleSheet.prototype,Y=Symbol(),Q=new WeakMap,R=class{constructor(e,t,s){if(this._$cssResult$=!0,s!==Y)throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");this.cssText=e,this.t=t}get styleSheet(){let e=this.o,t=this.t;if(N&&e===void 0){let s=t!==void 0&&t.length===1;s&&(e=Q.get(t)),e===void 0&&((this.o=e=new CSSStyleSheet).replaceSync(this.cssText),s&&Q.set(t,e))}return e}toString(){return this.cssText}},ee=i=>new R(typeof i=="string"?i:i+"",void 0,Y);var te=(i,e)=>{if(N)i.adoptedStyleSheets=e.map(t=>t instanceof CSSStyleSheet?t:t.styleSheet);else for(let t of e){let s=document.createElement("style"),o=U.litNonce;o!==void 0&&s.setAttribute("nonce",o),s.textContent=t.cssText,i.appendChild(s)}},B=N?i=>i:i=>i instanceof CSSStyleSheet?(e=>{let t="";for(let s of e.cssRules)t+=s.cssText;return ee(t)})(i):i;var{is:ye,defineProperty:be,getOwnPropertyDescriptor:ge,getOwnPropertyNames:me,getOwnPropertySymbols:$e,getPrototypeOf:Ae}=Object,O=globalThis,se=O.trustedTypes,we=se?se.emptyScript:"",_e=O.reactiveElementPolyfillSupport,x=(i,e)=>i,z={toAttribute(i,e){switch(e){case Boolean:i=i?we:null;break;case Object:case Array:i=i==null?i:JSON.stringify(i)}return i},fromAttribute(i,e){let t=i;switch(e){case Boolean:t=i!==null;break;case Number:t=i===null?null:Number(i);break;case Object:case Array:try{t=JSON.parse(i)}catch{t=null}}return t}},oe=(i,e)=>!ye(i,e),ie={attribute:!0,type:String,converter:z,reflect:!1,useDefault:!1,hasChanged:oe};Symbol.metadata??=Symbol("metadata"),O.litPropertyMetadata??=new WeakMap;var y=class extends HTMLElement{static addInitializer(e){this._$Ei(),(this.l??=[]).push(e)}static get observedAttributes(){return this.finalize(),this._$Eh&&[...this._$Eh.keys()]}static createProperty(e,t=ie){if(t.state&&(t.attribute=!1),this._$Ei(),this.prototype.hasOwnProperty(e)&&((t=Object.create(t)).wrapped=!0),this.elementProperties.set(e,t),!t.noAccessor){let s=Symbol(),o=this.getPropertyDescriptor(e,s,t);o!==void 0&&be(this.prototype,e,o)}}static getPropertyDescriptor(e,t,s){let{get:o,set:n}=ge(this.prototype,e)??{get(){return this[t]},set(r){this[t]=r}};return{get:o,set(r){let c=o?.call(this);n?.call(this,r),this.requestUpdate(e,c,s)},configurable:!0,enumerable:!0}}static getPropertyOptions(e){return this.elementProperties.get(e)??ie}static _$Ei(){if(this.hasOwnProperty(x("elementProperties")))return;let e=Ae(this);e.finalize(),e.l!==void 0&&(this.l=[...e.l]),this.elementProperties=new Map(e.elementProperties)}static finalize(){if(this.hasOwnProperty(x("finalized")))return;if(this.finalized=!0,this._$Ei(),this.hasOwnProperty(x("properties"))){let t=this.properties,s=[...me(t),...$e(t)];for(let o of s)this.createProperty(o,t[o])}let e=this[Symbol.metadata];if(e!==null){let t=litPropertyMetadata.get(e);if(t!==void 0)for(let[s,o]of t)this.elementProperties.set(s,o)}this._$Eh=new Map;for(let[t,s]of this.elementProperties){let o=this._$Eu(t,s);o!==void 0&&this._$Eh.set(o,t)}this.elementStyles=this.finalizeStyles(this.styles)}static finalizeStyles(e){let t=[];if(Array.isArray(e)){let s=new Set(e.flat(1/0).reverse());for(let o of s)t.unshift(B(o))}else e!==void 0&&t.push(B(e));return t}static _$Eu(e,t){let s=t.attribute;return s===!1?void 0:typeof s=="string"?s:typeof e=="string"?e.toLowerCase():void 0}constructor(){super(),this._$Ep=void 0,this.isUpdatePending=!1,this.hasUpdated=!1,this._$Em=null,this._$Ev()}_$Ev(){this._$ES=new Promise(e=>this.enableUpdating=e),this._$AL=new Map,this._$E_(),this.requestUpdate(),this.constructor.l?.forEach(e=>e(this))}addController(e){(this._$EO??=new Set).add(e),this.renderRoot!==void 0&&this.isConnected&&e.hostConnected?.()}removeController(e){this._$EO?.delete(e)}_$E_(){let e=new Map,t=this.constructor.elementProperties;for(let s of t.keys())this.hasOwnProperty(s)&&(e.set(s,this[s]),delete this[s]);e.size>0&&(this._$Ep=e)}createRenderRoot(){let e=this.shadowRoot??this.attachShadow(this.constructor.shadowRootOptions);return te(e,this.constructor.elementStyles),e}connectedCallback(){this.renderRoot??=this.createRenderRoot(),this.enableUpdating(!0),this._$EO?.forEach(e=>e.hostConnected?.())}enableUpdating(e){}disconnectedCallback(){this._$EO?.forEach(e=>e.hostDisconnected?.())}attributeChangedCallback(e,t,s){this._$AK(e,s)}_$ET(e,t){let s=this.constructor.elementProperties.get(e),o=this.constructor._$Eu(e,s);if(o!==void 0&&s.reflect===!0){let n=(s.converter?.toAttribute!==void 0?s.converter:z).toAttribute(t,s.type);this._$Em=e,n==null?this.removeAttribute(o):this.setAttribute(o,n),this._$Em=null}}_$AK(e,t){let s=this.constructor,o=s._$Eh.get(e);if(o!==void 0&&this._$Em!==o){let n=s.getPropertyOptions(o),r=typeof n.converter=="function"?{fromAttribute:n.converter}:n.converter?.fromAttribute!==void 0?n.converter:z;this._$Em=o;let c=r.fromAttribute(t,n.type);this[o]=c??this._$Ej?.get(o)??c,this._$Em=null}}requestUpdate(e,t,s,o=!1,n){if(e!==void 0){let r=this.constructor;if(o===!1&&(n=this[e]),s??=r.getPropertyOptions(e),!((s.hasChanged??oe)(n,t)||s.useDefault&&s.reflect&&n===this._$Ej?.get(e)&&!this.hasAttribute(r._$Eu(e,s))))return;this.C(e,t,s)}this.isUpdatePending===!1&&(this._$ES=this._$EP())}C(e,t,{useDefault:s,reflect:o,wrapped:n},r){s&&!(this._$Ej??=new Map).has(e)&&(this._$Ej.set(e,r??t??this[e]),n!==!0||r!==void 0)||(this._$AL.has(e)||(this.hasUpdated||s||(t=void 0),this._$AL.set(e,t)),o===!0&&this._$Em!==e&&(this._$Eq??=new Set).add(e))}async _$EP(){this.isUpdatePending=!0;try{await this._$ES}catch(t){Promise.reject(t)}let e=this.scheduleUpdate();return e!=null&&await e,!this.isUpdatePending}scheduleUpdate(){return this.performUpdate()}performUpdate(){if(!this.isUpdatePending)return;if(!this.hasUpdated){if(this.renderRoot??=this.createRenderRoot(),this._$Ep){for(let[o,n]of this._$Ep)this[o]=n;this._$Ep=void 0}let s=this.constructor.elementProperties;if(s.size>0)for(let[o,n]of s){let{wrapped:r}=n,c=this[o];r!==!0||this._$AL.has(o)||c===void 0||this.C(o,void 0,n,c)}}let e=!1,t=this._$AL;try{e=this.shouldUpdate(t),e?(this.willUpdate(t),this._$EO?.forEach(s=>s.hostUpdate?.()),this.update(t)):this._$EM()}catch(s){throw e=!1,this._$EM(),s}e&&this._$AE(t)}willUpdate(e){}_$AE(e){this._$EO?.forEach(t=>t.hostUpdated?.()),this.hasUpdated||(this.hasUpdated=!0,this.firstUpdated(e)),this.updated(e)}_$EM(){this._$AL=new Map,this.isUpdatePending=!1}get updateComplete(){return this.getUpdateComplete()}getUpdateComplete(){return this._$ES}shouldUpdate(e){return!0}update(e){this._$Eq&&=this._$Eq.forEach(t=>this._$ET(t,this[t])),this._$EM()}updated(e){}firstUpdated(e){}};y.elementStyles=[],y.shadowRootOptions={mode:"open"},y[x("elementProperties")]=new Map,y[x("finalized")]=new Map,_e?.({ReactiveElement:y}),(O.reactiveElementVersions??=[]).push("2.1.2");var V=globalThis,ae=i=>i,L=V.trustedTypes,ne=L?L.createPolicy("lit-html",{createHTML:i=>i}):void 0,he="$lit$",g=`lit$${Math.random().toFixed(9).slice(2)}$`,ue="?"+g,Se=`<${ue}>`,A=document,k=()=>A.createComment(""),C=i=>i===null||typeof i!="object"&&typeof i!="function",q=Array.isArray,Ee=i=>q(i)||typeof i?.[Symbol.iterator]=="function",I=`[ 	
\f\r]`,P=/<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g,re=/-->/g,le=/>/g,m=RegExp(`>|${I}(?:([^\\s"'>=/]+)(${I}*=${I}*(?:[^ 	
\f\r"'\`<>=]|("|')|))|$)`,"g"),de=/'/g,ce=/"/g,fe=/^(?:script|style|textarea|title)$/i,Z=i=>(e,...t)=>({_$litType$:i,strings:e,values:t}),f=Z(1),We=Z(2),Ge=Z(3),w=Symbol.for("lit-noChange"),h=Symbol.for("lit-nothing"),pe=new WeakMap,$=A.createTreeWalker(A,129);function ve(i,e){if(!q(i)||!i.hasOwnProperty("raw"))throw Error("invalid template strings array");return ne!==void 0?ne.createHTML(e):e}var xe=(i,e)=>{let t=i.length-1,s=[],o,n=e===2?"<svg>":e===3?"<math>":"",r=P;for(let c=0;c<t;c++){let l=i[c],p,u,d=-1,v=0;for(;v<l.length&&(r.lastIndex=v,u=r.exec(l),u!==null);)v=r.lastIndex,r===P?u[1]==="!--"?r=re:u[1]!==void 0?r=le:u[2]!==void 0?(fe.test(u[2])&&(o=RegExp("</"+u[2],"g")),r=m):u[3]!==void 0&&(r=m):r===m?u[0]===">"?(r=o??P,d=-1):u[1]===void 0?d=-2:(d=r.lastIndex-u[2].length,p=u[1],r=u[3]===void 0?m:u[3]==='"'?ce:de):r===ce||r===de?r=m:r===re||r===le?r=P:(r=m,o=void 0);let b=r===m&&i[c+1].startsWith("/>")?" ":"";n+=r===P?l+Se:d>=0?(s.push(p),l.slice(0,d)+he+l.slice(d)+g+b):l+g+(d===-2?c:b)}return[ve(i,n+(i[t]||"<?>")+(e===2?"</svg>":e===3?"</math>":"")),s]},T=class i{constructor({strings:e,_$litType$:t},s){let o;this.parts=[];let n=0,r=0,c=e.length-1,l=this.parts,[p,u]=xe(e,t);if(this.el=i.createElement(p,s),$.currentNode=this.el.content,t===2||t===3){let d=this.el.content.firstChild;d.replaceWith(...d.childNodes)}for(;(o=$.nextNode())!==null&&l.length<c;){if(o.nodeType===1){if(o.hasAttributes())for(let d of o.getAttributeNames())if(d.endsWith(he)){let v=u[r++],b=o.getAttribute(d).split(g),M=/([.?@])?(.*)/.exec(v);l.push({type:1,index:n,name:M[2],strings:b,ctor:M[1]==="."?j:M[1]==="?"?W:M[1]==="@"?G:S}),o.removeAttribute(d)}else d.startsWith(g)&&(l.push({type:6,index:n}),o.removeAttribute(d));if(fe.test(o.tagName)){let d=o.textContent.split(g),v=d.length-1;if(v>0){o.textContent=L?L.emptyScript:"";for(let b=0;b<v;b++)o.append(d[b],k()),$.nextNode(),l.push({type:2,index:++n});o.append(d[v],k())}}}else if(o.nodeType===8)if(o.data===ue)l.push({type:2,index:n});else{let d=-1;for(;(d=o.data.indexOf(g,d+1))!==-1;)l.push({type:7,index:n}),d+=g.length-1}n++}}static createElement(e,t){let s=A.createElement("template");return s.innerHTML=e,s}};function _(i,e,t=i,s){if(e===w)return e;let o=s!==void 0?t._$Co?.[s]:t._$Cl,n=C(e)?void 0:e._$litDirective$;return o?.constructor!==n&&(o?._$AO?.(!1),n===void 0?o=void 0:(o=new n(i),o._$AT(i,t,s)),s!==void 0?(t._$Co??=[])[s]=o:t._$Cl=o),o!==void 0&&(e=_(i,o._$AS(i,e.values),o,s)),e}var F=class{constructor(e,t){this._$AV=[],this._$AN=void 0,this._$AD=e,this._$AM=t}get parentNode(){return this._$AM.parentNode}get _$AU(){return this._$AM._$AU}u(e){let{el:{content:t},parts:s}=this._$AD,o=(e?.creationScope??A).importNode(t,!0);$.currentNode=o;let n=$.nextNode(),r=0,c=0,l=s[0];for(;l!==void 0;){if(r===l.index){let p;l.type===2?p=new H(n,n.nextSibling,this,e):l.type===1?p=new l.ctor(n,l.name,l.strings,this,e):l.type===6&&(p=new K(n,this,e)),this._$AV.push(p),l=s[++c]}r!==l?.index&&(n=$.nextNode(),r++)}return $.currentNode=A,o}p(e){let t=0;for(let s of this._$AV)s!==void 0&&(s.strings!==void 0?(s._$AI(e,s,t),t+=s.strings.length-2):s._$AI(e[t])),t++}},H=class i{get _$AU(){return this._$AM?._$AU??this._$Cv}constructor(e,t,s,o){this.type=2,this._$AH=h,this._$AN=void 0,this._$AA=e,this._$AB=t,this._$AM=s,this.options=o,this._$Cv=o?.isConnected??!0}get parentNode(){let e=this._$AA.parentNode,t=this._$AM;return t!==void 0&&e?.nodeType===11&&(e=t.parentNode),e}get startNode(){return this._$AA}get endNode(){return this._$AB}_$AI(e,t=this){e=_(this,e,t),C(e)?e===h||e==null||e===""?(this._$AH!==h&&this._$AR(),this._$AH=h):e!==this._$AH&&e!==w&&this._(e):e._$litType$!==void 0?this.$(e):e.nodeType!==void 0?this.T(e):Ee(e)?this.k(e):this._(e)}O(e){return this._$AA.parentNode.insertBefore(e,this._$AB)}T(e){this._$AH!==e&&(this._$AR(),this._$AH=this.O(e))}_(e){this._$AH!==h&&C(this._$AH)?this._$AA.nextSibling.data=e:this.T(A.createTextNode(e)),this._$AH=e}$(e){let{values:t,_$litType$:s}=e,o=typeof s=="number"?this._$AC(e):(s.el===void 0&&(s.el=T.createElement(ve(s.h,s.h[0]),this.options)),s);if(this._$AH?._$AD===o)this._$AH.p(t);else{let n=new F(o,this),r=n.u(this.options);n.p(t),this.T(r),this._$AH=n}}_$AC(e){let t=pe.get(e.strings);return t===void 0&&pe.set(e.strings,t=new T(e)),t}k(e){q(this._$AH)||(this._$AH=[],this._$AR());let t=this._$AH,s,o=0;for(let n of e)o===t.length?t.push(s=new i(this.O(k()),this.O(k()),this,this.options)):s=t[o],s._$AI(n),o++;o<t.length&&(this._$AR(s&&s._$AB.nextSibling,o),t.length=o)}_$AR(e=this._$AA.nextSibling,t){for(this._$AP?.(!1,!0,t);e!==this._$AB;){let s=ae(e).nextSibling;ae(e).remove(),e=s}}setConnected(e){this._$AM===void 0&&(this._$Cv=e,this._$AP?.(e))}},S=class{get tagName(){return this.element.tagName}get _$AU(){return this._$AM._$AU}constructor(e,t,s,o,n){this.type=1,this._$AH=h,this._$AN=void 0,this.element=e,this.name=t,this._$AM=o,this.options=n,s.length>2||s[0]!==""||s[1]!==""?(this._$AH=Array(s.length-1).fill(new String),this.strings=s):this._$AH=h}_$AI(e,t=this,s,o){let n=this.strings,r=!1;if(n===void 0)e=_(this,e,t,0),r=!C(e)||e!==this._$AH&&e!==w,r&&(this._$AH=e);else{let c=e,l,p;for(e=n[0],l=0;l<n.length-1;l++)p=_(this,c[s+l],t,l),p===w&&(p=this._$AH[l]),r||=!C(p)||p!==this._$AH[l],p===h?e=h:e!==h&&(e+=(p??"")+n[l+1]),this._$AH[l]=p}r&&!o&&this.j(e)}j(e){e===h?this.element.removeAttribute(this.name):this.element.setAttribute(this.name,e??"")}},j=class extends S{constructor(){super(...arguments),this.type=3}j(e){this.element[this.name]=e===h?void 0:e}},W=class extends S{constructor(){super(...arguments),this.type=4}j(e){this.element.toggleAttribute(this.name,!!e&&e!==h)}},G=class extends S{constructor(e,t,s,o,n){super(e,t,s,o,n),this.type=5}_$AI(e,t=this){if((e=_(this,e,t,0)??h)===w)return;let s=this._$AH,o=e===h&&s!==h||e.capture!==s.capture||e.once!==s.once||e.passive!==s.passive,n=e!==h&&(s===h||o);o&&this.element.removeEventListener(this.name,this,s),n&&this.element.addEventListener(this.name,this,e),this._$AH=e}handleEvent(e){typeof this._$AH=="function"?this._$AH.call(this.options?.host??this.element,e):this._$AH.handleEvent(e)}},K=class{constructor(e,t,s){this.element=e,this.type=6,this._$AN=void 0,this._$AM=t,this.options=s}get _$AU(){return this._$AM._$AU}_$AI(e){_(this,e)}};var Pe=V.litHtmlPolyfillSupport;Pe?.(T,H),(V.litHtmlVersions??=[]).push("3.3.3");var D=(i,e,t)=>{let s=t?.renderBefore??e,o=s._$litPart$;if(o===void 0){let n=t?.renderBefore??null;s._$litPart$=o=new H(e.insertBefore(k(),n),n,void 0,t??{})}return o._$AI(i),o};var X=globalThis,E=class extends y{constructor(){super(...arguments),this.renderOptions={host:this},this._$Do=void 0}createRenderRoot(){let e=super.createRenderRoot();return this.renderOptions.renderBefore??=e.firstChild,e}update(e){let t=this.render();this.hasUpdated||(this.renderOptions.isConnected=this.isConnected),super.update(e),this._$Do=D(t,this.renderRoot,this.renderOptions)}connectedCallback(){super.connectedCallback(),this._$Do?.setConnected(!0)}disconnectedCallback(){super.disconnectedCallback(),this._$Do?.setConnected(!1)}render(){return w}};E._$litElement$=!0,E.finalized=!0,X.litElementHydrateSupport?.({LitElement:E});var ke=X.litElementPolyfillSupport;ke?.({LitElement:E});(X.litElementVersions??=[]).push("4.2.2");function a(i,...e){let t=window[i];if(typeof t!="function")throw new Error(`UI action ${String(i)} is not available`);t(...e)}function Ce(i){i.key!=="Enter"&&i.key!==" "||(i.preventDefault(),a("toggleAnalysisSheet",i))}var Te=()=>f`
  <div id="topbar">
    <h1><span class="mark">He<span class="mark-x">X</span>O</span><span class="mark-tag">Observatory</span></h1>
    <span id="view-label">Analysis</span>
    <span id="status">Loading…</span>
    <span id="difficulty-badge" hidden></span>
    <button id="resign-btn" hidden @click=${()=>a("resign")}>Resign</button>
    <button id="copy-htttx-btn" hidden @click=${()=>a("copyHtttx")}>Copy game record</button>
    <button id="analyze-game-btn" hidden @click=${()=>a("analyzeThisGame")}>Analyze this game</button>
    <button id="new-game-btn" @click=${()=>a("openModal")}>New game</button>
    <button id="analysis-btn" @click=${()=>a("goToAnalysis")}>Analysis</button>
    <button id="play-btn" @click=${()=>a("goToPlay")}>&larr; Play</button>
    <button id="topbar-menu-btn" type="button" aria-haspopup="true" aria-expanded="false"
      aria-label="Open menu" @click=${i=>a("toggleTopbarMenu",i)}>
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
`,He=()=>f`
  <div id="analysis-controls">
    <div id="analysis-sheet-handle" data-label="Controls" aria-label="Toggle controls panel"
      role="button" tabindex="0" @click=${i=>a("toggleAnalysisSheet",i)}
      @keydown=${Ce}></div>
    <div class="analysis-mode-tabs" role="tablist" aria-label="Analysis tools">
      <button id="analysis-mode-analysis" type="button" role="tab" aria-selected="true"
        aria-controls="analysis-controls-body" @click=${()=>a("closeProofLab")}>Analysis</button>
      <button id="proof-lab-launch" type="button" role="tab" aria-selected="false"
        aria-controls="proof-lab-drawer" disabled @click=${()=>a("openProofLab")}>
        <span class="proof-lab-launch-icon" aria-hidden="true">◇</span> Proof lab
      </button>
    </div>
    <div id="analysis-info"></div>
    <div id="analysis-position-browser" aria-label="Position navigation">
      <div id="analysis-eval-wrap" hidden>
        <canvas id="analysis-eval-bar" width="320" height="48" tabindex="0" role="slider"
          aria-label="Game position timeline" aria-valuemin="1" aria-valuemax="1" aria-valuenow="1"
          @pointerdown=${i=>a("onAnalysisEvalPointerDown",i)}
          @pointermove=${i=>a("onAnalysisEvalPointerMove",i)}
          @pointerup=${i=>a("onAnalysisEvalPointerUp",i)}
          @pointercancel=${i=>a("onAnalysisEvalPointerUp",i)}
          @pointerleave=${()=>a("onAnalysisEvalPointerLeave")}
          @click=${i=>a("onAnalysisEvalClick",i)}
          @keydown=${i=>a("onAnalysisEvalKeydown",i)}></canvas>
        <div id="analysis-eval-preview" hidden></div>
      </div>
      <div id="analysis-navigation" class="row" hidden>
        <button id="analysis-previous-position" class="analysis-nav-icon" type="button"
          aria-label="Previous position" title="Previous position" @click=${()=>a("analysisUndo")}>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M9 4 3 10l6 6M4 10h13"/></svg>
        </button>
        <button id="analysis-latest-mainline" class="analysis-nav-icon" type="button"
          aria-label="Latest position in the game" title="Latest position in the game"
          @click=${()=>a("returnToMainline")}>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 4 6 6-6 6M14 4v12"/></svg>
        </button>
      </div>
    </div>
    <div id="analysis-controls-body" role="tabpanel" aria-labelledby="analysis-mode-analysis">
      <div id="analysis-source-summary" hidden>
        <span><strong id="analysis-source-title">Loaded game</strong><small id="analysis-source-meta"></small></span>
        <span class="analysis-source-actions">
          <button id="analysis-copy-htttx" type="button" class="secondary-button" @click=${()=>a("copyAnalysisHtttx")}>Copy as HTTTX</button>
          <button type="button" class="secondary-button" @click=${()=>a("editAnalysisSource")}>Change</button>
        </span>
      </div>
      <div id="analysis-setup">
        <button id="hds-import-trigger" type="button" @click=${()=>a("openHdsImport")}>
          Import from Hexo sandbox <span aria-hidden="true">&rarr;</span>
        </button>
        <label class="analysis-record-field" for="analysis-htttx">
          <span class="field-label">Paste a game record (HTTTX)</span>
          <span id="analysis-record-hint" class="field-hint">Open the game now. Analysis starts only when you ask for it.</span>
          <textarea id="analysis-htttx" rows="4" placeholder="version[1];\n1. [1,0][2,0];\n..."
            aria-describedby="analysis-record-hint"
            @input=${()=>a("analysisInputChanged")}></textarea>
        </label>
        <div class="analysis-setup-actions">
          <button id="analysis-source-cancel" type="button" class="secondary-button" hidden
            @click=${()=>a("cancelAnalysisSourceEdit")}>Cancel</button>
          <button id="analysis-load-btn" class="primary-button" @click=${()=>a("loadGame")}>Load game</button>
        </div>
      </div>
      <div class="analysis-run-group">
        <div class="analysis-run-actions">
          <button id="analysis-position-btn" @click=${()=>a("analyzeCurrentPosition")} disabled>Analyze position</button>
          <button id="analysis-game-btn" @click=${()=>a("analyzeWholeGame")} disabled>Analyze full game</button>
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
            <label id="analysis-model-field" class="analysis-strength-field" for="analysis-model" hidden>
              <span class="field-label">Strix version</span>
              <span class="field-hint">Choose which trained model evaluates this position.</span>
              <select id="analysis-model" @change=${()=>a("selectAnalysisModel")}></select>
            </label>
            <label class="analysis-strength-field" for="analysis-strength">
              <span class="field-label">Analysis effort</span>
              <span id="analysis-strength-hint" class="field-hint">Higher settings examine more possible continuations and take longer. Instant gives an estimate without searching ahead.</span>
              <select id="analysis-strength" aria-describedby="analysis-strength-hint" @change=${()=>a("saveAnalysisStrength")}>
                <option value="network">Instant · no search</option>
                <option value="quick">Quick</option>
                <option value="standard" selected>Standard</option>
                <option value="strong">Strong</option>
                <option value="deep">Deep</option>
              </select>
            </label>
            <label class="analysis-setting-toggle" for="analysis-auto-branch">
              <input id="analysis-auto-branch" type="checkbox"
                @change=${()=>a("saveAutomaticAnalysis")}>
              <span><strong>Analyze new moves automatically</strong><small>Start analysis after you place a hex</small></span>
            </label>
            <label class="analysis-setting-toggle" for="analysis-auto-forcing">
              <input id="analysis-auto-forcing" type="checkbox" checked
                @change=${()=>a("saveAutomaticForcing")}>
              <span><strong>Check for forced wins</strong><small>Look for a win the opponent cannot stop</small></span>
            </label>
          </section>
          <section class="analysis-settings-section" aria-labelledby="analysis-display-settings-title">
            <h3 id="analysis-display-settings-title">Board overlays</h3>
            <div class="analysis-display-options-body">
              <label><input type="checkbox" id="analysis-heatmap" checked @change=${()=>a("saveDisplayPreferences")}> Suggested moves</label>
              <label><input type="checkbox" id="analysis-forcing" checked @change=${()=>a("saveDisplayPreferences")}> Winning lines</label>
              <label><input type="checkbox" id="analysis-threats" @change=${()=>a("saveDisplayPreferences")}> Threats to answer</label>
            </div>
          </section>
          <section class="analysis-settings-section" aria-labelledby="analysis-numbering-settings-title">
            <h3 id="analysis-numbering-settings-title">Position numbering</h3>
            <label class="analysis-strength-field" for="analysis-numbering">
              <span class="field-label">Numbering</span>
              <span class="field-hint">Number each placement, or group each full round of both players.</span>
              <select id="analysis-numbering" @change=${()=>a("savePositionNumbering")}>
                <option value="ply" selected>Ply (1, 2, 3…)</option>
                <option value="round">Round (1, 1, 1, 2…)</option>
              </select>
            </label>
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
`,Me=()=>f`
  <dialog id="hds-import-dialog" aria-labelledby="hds-import-title"
    @click=${i=>{i.target===i.currentTarget&&a("closeHdsImport")}}>
    <form @submit=${i=>{i.preventDefault(),a("convertHds")}}>
      <header class="hds-dialog-header">
        <div>
          <h2 id="hds-import-title">Import from Hexo sandbox</h2>
          <p>Paste the position's hexo.did.science link or short code.</p>
        </div>
        <button class="dialog-close" type="button" aria-label="Close import dialog"
          @click=${()=>a("closeHdsImport")}>Close</button>
      </header>
      <label for="hds-input"><span class="field-label">Sandbox link or code</span>
        <input id="hds-input" type="text" inputmode="url" autocomplete="off"
          placeholder="https://hexo.did.science/sandbox/5knldz6">
      </label>
      <div id="hds-status" role="status" aria-live="polite"></div>
      <footer class="hds-dialog-actions">
        <button class="secondary-button" type="button" @click=${()=>a("closeHdsImport")}>Cancel</button>
        <button class="primary-button" type="submit">Import position</button>
      </footer>
    </form>
  </dialog>
`,Ue=()=>f`
  <div id="analysis-forcing-depth-control" class="proof-lab-form">
    <section class="proof-lab-settings" aria-labelledby="proof-search-settings-title">
      <h3 id="proof-search-settings-title">Search settings</h3>
      <label for="analysis-forcing-engine">Search method
        <span id="analysis-forcing-engine-hint" class="field-hint">The default first proves a win, then rules out every shorter win. It saves all checked replies and the best-defence line.</span>
        <select id="analysis-forcing-engine" aria-describedby="analysis-forcing-engine-hint" @change=${()=>a("updateForcingSolverUi")}>
          <option value="pdspn-shortest" selected>Prove the shortest win · PDS-PN</option>
          <option value="pdspn">Find and explore a win · PDS-PN</option>
          <option value="idtt">Bounded shortest check · IDTT</option>
        </select>
      </label>
      <label for="analysis-forcing-width">Moves to consider
        <select id="analysis-forcing-width">
          <option value="wide" selected>Broad · all legal moves</option>
          <option value="tight">Direct only · immediate threats</option>
        </select>
      </label>
      <label id="analysis-forcing-depth-row" for="analysis-forcing-depth">
        <span id="analysis-forcing-depth-label">Longest win to check</span>
        <span class="analysis-depth-input"><input id="analysis-forcing-depth" type="number" min="1" max="60" value="25" step="1" inputmode="numeric"> turns</span>
      </label>
      <label id="analysis-forcing-effort-row" for="analysis-forcing-effort">
        <span class="proof-effort-heading"><span>Search effort</span><output id="analysis-forcing-effort-label" for="analysis-forcing-effort">Standard</output></span>
        <input id="analysis-forcing-effort" type="range" min="0" max="5" value="1" step="1"
          aria-describedby="analysis-forcing-effort-hint" @input=${()=>a("updateForcingEffortUi")}>
        <span id="analysis-forcing-effort-hint" class="field-hint">Good default for most positions.</span>
      </label>
    </section>
    <div class="analysis-solver-actions">
      <button id="analysis-solve-forcing-btn" @click=${()=>a("solveCurrentForcing")}>Check for a forced win</button>
      <button id="analysis-cancel-forcing-btn" @click=${()=>a("cancelForcingSolve")} hidden>Stop search</button>
      <button id="analysis-explore-certificate-btn" @click=${()=>a("openProofExplorer")} hidden>View all replies</button>
      <button id="analysis-share-certificate-btn" @click=${()=>a("shareForcingCertificate")} hidden>Copy result link</button>
      <button id="analysis-download-certificate-btn" @click=${()=>a("downloadForcingCertificate")} hidden>Download result</button>
      <span id="proof-share-status" class="proof-share-status" role="status" aria-live="polite"></span>
    </div>
    <div id="analysis-forcing-status" role="status" aria-live="polite">Ready. This search runs on your device.</div>
    <details class="analysis-advanced proof-lab-help">
      <summary>How the search works</summary>
      <div class="proof-lab-help-body">
        <p id="analysis-solver-help">First finds and verifies a forced win. Then reuses that proof to rule out every shorter win. The saved best-defence line shows the replies that delay the win longest.</p>
        <p>Search effort controls how long the solver may keep trying. PDS-PN automatically races several complementary branch strategies. Broad search considers every legal move; direct-only search is faster but considers only immediate threats.</p>
      </div>
    </details>
  </div>
`,Re=()=>f`
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
    <section id="proof-defence-review" class="proof-defence-review" aria-labelledby="proof-defence-review-title">
      <div>
        <h3 id="proof-defence-review-title">How could I have defended?</h3>
        <p id="proof-defence-review-copy">Walk back through this lost replay and find the latest defence that breaks or delays the forced win.</p>
      </div>
      <div class="proof-defence-review-actions">
        <button id="proof-find-defence-btn" class="secondary-button" type="button" @click=${()=>a("findBetterDefence")}>Find a better defence</button>
        <button id="proof-stop-defence-btn" class="secondary-button" type="button" hidden @click=${()=>a("cancelBetterDefence")}>Stop</button>
      </div>
      <div id="proof-defence-status" class="proof-defence-status" role="status" aria-live="polite"></div>
      <div id="proof-defence-result" class="proof-defence-result" hidden></div>
    </section>
    ${Ue()}
  </aside>
`,Ne=()=>f`
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
`,Oe=()=>f`
  <div id="proof-explorer" role="dialog" aria-modal="true" aria-labelledby="proof-explorer-title" hidden>
    <section id="proof-board-container" aria-label="Proof position board">
      <svg id="proof-board" aria-label="HeXO proof board"></svg>
      <div class="proof-explorer-actions" aria-label="Proof explorer actions">
        <button id="proof-share-btn" @click=${()=>a("shareForcingCertificate")}
          title="Save this result and copy its link">Copy link</button>
        <button @click=${()=>a("downloadForcingCertificate")}>Download</button>
        <button id="proof-close-btn" class="proof-close" @click=${()=>a("closeProofExplorer")}
          aria-label="Close proof explorer">Close</button>
      </div>
      <div class="proof-board-tools" aria-label="Proof board tools">
        <label class="proof-board-toggle" for="proof-show-line">
          <input id="proof-show-line" type="checkbox"
            @change=${i=>a("proofSetShowLine",i.target.checked)}>
          <span>Show winning line</span>
        </label>
        <div class="proof-board-zoom" aria-label="Board zoom controls">
          <button @click=${()=>a("proofZoom",1.25)} aria-label="Zoom in">+</button>
          <button @click=${()=>a("proofZoom",.8)} aria-label="Zoom out">−</button>
          <button @click=${()=>a("proofFitBoard")}>Fit</button>
        </div>
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
          <button id="proof-back-btn" @click=${()=>a("proofExplorerBack")} title="Go back one step">&larr; Back</button>
          <button @click=${()=>a("proofExplorerReset")} title="Return to the first position">Start again</button>
        </nav>
        <div class="proof-progress-copy"><span id="proof-progress-label"></span><span id="proof-node-label"></span></div>
        <div class="proof-progress-track"><div id="proof-progress-bar"></div></div>
        <div id="proof-optimization-note" class="proof-optimization-note" hidden></div>
        <div id="proof-step-card"></div>
        <div class="proof-path-heading"><span>Proof path</span><small><span class="proof-hover-hint">hover to preview · </span>choose to follow</small></div>
        <div id="proof-tree" class="proof-tree" role="tree" aria-label="Positions and available branches"></div>
        <div class="proof-panel-actions">
          <button id="proof-shortest-line-btn" @click=${()=>a("proofExplorerToggleShortestLine")} hidden>Longest defence</button>
          <button id="proof-worst-btn" class="proof-primary" @click=${()=>a("proofExplorerWorstCase")}
            title="Follow the reply that delays the win longest">Choose longest defence &rarr;</button>
        </div>
        <details class="proof-explorer-note">
          <summary>How to read this proof</summary>
          <p>On the winning side's turn, each branch shown is a move that this search proved will win. On the other side's turn, every checked reply is shown. “Longest defence” follows the reply that delays the win for the most turns.</p>
        </details>
      </aside>
    </section>
  </div>
`,Le=()=>f`
  <div id="modal-bg">
    <div id="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <h2 id="modal-title">New game</h2>
      <label id="play-model-field" for="play-model" hidden>
        <span class="field-label">Opponent</span>
        <span class="field-optional">Choose a Strix version</span>
        <select id="play-model" @change=${()=>a("selectPlayModel")}></select>
      </label>
      <div id="bot-stats">
        <div id="bot-stats-current">Loading the bot's record…</div>
        <div id="bot-stats-alltime"></div>
      </div>
      <label for="modal-name"><span class="field-label">Name</span><span class="field-optional">Optional</span><input id="modal-name" type="text" maxlength="64" autocomplete="off"></label>
      <label for="modal-elo"><span class="field-label">Your rating (Elo)</span><span class="field-optional">Optional · enter your own estimate</span><input id="modal-elo" type="number" min="0" max="3500" placeholder="1500" autocomplete="off" inputmode="numeric"></label>
      <fieldset class="side-fieldset">
        <legend>Side</legend>
        <div class="side-row">
          <button class="side-btn" data-side="P1" @click=${()=>a("selectSide","P1")}><span class="stone stone-p1">●</span>P1 <span class="side-colour">orange</span></button>
          <button class="side-btn selected" data-side="random" @click=${()=>a("selectSide","random")}><span class="stone">?</span>Random</button>
          <button class="side-btn" data-side="P2" @click=${()=>a("selectSide","P2")}><span class="stone stone-p2">●</span>P2 <span class="side-colour">blue</span></button>
        </div>
      </fieldset>
      <label id="diff-label" hidden>Search effort</label>
      <div id="diff-row" class="diff-row" hidden></div>
      <button id="start-btn" @click=${()=>a("startGame")}>Start game</button>
    </div>
  </div>
`,De=()=>f`
  ${Te()}
  <div id="board-container"><svg id="board"></svg></div>
  ${Ne()}
  ${Oe()}
  ${Me()}
  ${Le()}
`,J=class extends HTMLElement{connectedCallback(){D(De(),this)}};customElements.define("hexo-observatory-app",J);})();
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
