const q={good:{icon:"✓",label:"Good",color:"#79cf9a"},bad:{icon:"×",label:"Mistake",color:"#e25c5c"},
  best:{icon:"★",label:"Best",color:"#f2c14e"},unclear:{icon:"◇",label:"Unclear",color:"#8b8f88"}};
const pairs=[
  [[[2,2],[3,3]],[[-3,4],[1,4]]], [[[4,4],[5,5]],[[2,5],[0,6]]],
  [[[6,6],[7,7]],[[3,6],[4,7]]], [[[5,10],[6,11]],[[5,8],[3,11]]],
  [[[-1,1],[1,-1]],[[-3,3],[2,-2]]], [[[1,0],[1,-2]],[[1,2],[1,-3]]],
  [[[2,0],[3,0]],[[-1,0],[4,0]]], [[[3,1],[3,2]],[[3,4],[3,-1]]],
  [[[2,1],[5,1]],[[6,1],[0,1]]], [[[4,2],[6,0]],[[2,4],[7,-1]]],
];
let id=1;
const rounds=pairs.map((pair,index)=>({number:index+3,
  P1:pair[0].map(([qv,rv],slot)=>({id:id++,q:qv,r:rv,quality:slot===1?(index%3===0?q.good:index%3===1?q.bad:q.best):null,
    missedWin:(index===1&&slot===0)||(index===2&&slot===1)||(index===3&&slot===0)})),
  P2:pair[1].map(([qv,rv],slot)=>({id:id++,q:qv,r:rv,quality:slot===1?(index%2?q.good:q.bad):null,
    missedWin:(index===0&&slot===1)||(index===2&&slot===0)}))}));
rounds.push({number:13,P1:[{id:id++,q:-10,r:8,quality:q.unclear}],P2:[]});
const list=document.getElementById("preview-list"); list.rounds=rounds; list.activeId="18";
list.addEventListener("move-select",event=>{list.activeId=String(event.detail.id)});
list.addEventListener("missed-win-select",event=>console.log("missed win",event.detail.id));
