// Independent double-precision integration reference for correctness tests only.
function normalCDF(x){if(x>12)return 1;if(x<-12)return 0;const end=Math.abs(x),steps=256,h=end/steps;let sum=1+Math.exp(-end*end/2);for(let i=1;i<steps;i++)sum+=(i%2?4:2)*Math.exp(-((i*h)**2)/2);return 0.5+Math.sign(x)*h*sum/(3*Math.sqrt(2*Math.PI));}
export function optionReference(S,X,T,R,V){const d1=(Math.log(S/X)+(R+V*V/2)*T)/(V*Math.sqrt(T)),d2=d1-V*Math.sqrt(T),discount=X*Math.exp(-R*T);return [S*normalCDF(d1)-discount*normalCDF(d2),discount*normalCDF(-d2)-S*normalCDF(-d1)];}
export function blackScholesFixture(optN=256,guards=0){
 const count=Math.ceil(optN/2)*2+guards,d_StockPrice=Float32Array.from({length:count},(_,i)=>5+(i*7%397)),d_OptionStrike=Float32Array.from({length:count},(_,i)=>1+(i*11%431)),d_OptionYears=Float32Array.from({length:count},(_,i)=>0.03125+(i%64)/8),d_CallResult=new Float32Array(count).fill(-12345),d_PutResult=d_CallResult.slice(),calls=Array.from(d_CallResult),puts=Array.from(d_PutResult),scalars={Riskfree:Math.fround(0.02),Volatility:Math.fround(0.3),optN};
 for(let i=0;i<Math.floor(optN/2)*2;i++)[calls[i],puts[i]]=optionReference(d_StockPrice[i],d_OptionStrike[i],d_OptionYears[i],scalars.Riskfree,scalars.Volatility);
 return {buffers:{d_CallResult,d_PutResult,d_StockPrice,d_OptionStrike,d_OptionYears},scalars,out:'d_CallResult',expected:calls,expectedOutputs:{d_CallResult:calls,d_PutResult:puts},groups:[Math.max(1,Math.ceil(optN/2/128)),1,1],absoluteTolerance:0.0002,relativeTolerance:0.00002};
}
